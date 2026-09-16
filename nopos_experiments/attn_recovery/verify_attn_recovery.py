"""Verify the self-attention recovery setup.

Checks the claims the experiment rests on, in two parts.

Part A needs nothing but the code -- it pins Q/K to zero so every attention logit is
0 and the output is the plain mean of the visible positions, which is known in closed
form: prefix mean for C, suffix mean for F, global mean for B. It then confirms the
masks really block information, by perturbing one position and checking which outputs
move.

Part B needs a trained teacher and the binarised data. It builds the student exactly
as the training script does and confirms the teacher is frozen, stays bit-identical
across student updates, and that the regression actually optimises.

Usage:
  # Part A only
  python nopos_experiments/attn_recovery/verify_attn_recovery.py

  # Part A + B
  python nopos_experiments/attn_recovery/verify_attn_recovery.py \
      --teacher-checkpoint checkpoints_attn_recovery/teacher_seed1/checkpoint_last.pt \
      --data data-bin/wikitext-103
"""

import argparse
import math
import sys
from argparse import Namespace

import torch

from fairseq import checkpoint_utils, models, tasks
from fairseq.models.simple_attn_recovery import (
    FullWidthMaskedAttention,
    SimpleAttnTeacherModel,
    parse_mask_spec,
)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    return bool(ok)


def uniform_attention(spec, d):
    """Attention with Q/K pinned to zero, V and out_proj identity.

    Mirrors the pin in ``fairseq/models/fixed_attn_lm.py``: with W_q = W_k = 0 every
    logit is 0, so the softmax over the visible keys is exactly uniform and the output
    is their mean. Averaging the heads in out_proj makes the expected value closed
    form for multi-head specs too.
    """
    attn = FullWidthMaskedAttention(d, spec, dropout=0.0)
    with torch.no_grad():
        for h in range(attn.num_heads):
            attn.q_proj[h].weight.zero_()
            attn.q_proj[h].bias.zero_()
            attn.k_proj[h].weight.zero_()
            attn.k_proj[h].bias.zero_()
            attn.v_proj[h].weight.copy_(torch.eye(d))
            attn.v_proj[h].bias.zero_()
        attn.out_proj.weight.copy_(
            torch.cat([torch.eye(d)] * attn.num_heads, dim=1) / attn.num_heads
        )
    return attn.eval()


def part_a(seq_len=6, dim=8):
    torch.manual_seed(0)
    B, T, d = 2, seq_len, dim
    x = torch.randn(B, T, d)

    prefix = x.cumsum(dim=1) / torch.arange(1, T + 1).view(1, T, 1)
    suffix = x.flip(1).cumsum(dim=1).flip(1) / torch.arange(T, 0, -1).view(1, T, 1)
    glob = x.mean(dim=1, keepdim=True).expand(B, T, d)

    print("\n[A1] attention equals the mean over visible positions")
    with torch.no_grad():
        out_c = uniform_attention("C", d)(x)
        out_f = uniform_attention("F", d)(x)
        out_b = uniform_attention("B", d)(x)
        out_cf = uniform_attention("C,F", d)(x)
    check("C = prefix mean", torch.allclose(out_c, prefix, atol=1e-5),
          f"max err {(out_c - prefix).abs().max():.2e}")
    check("F = suffix mean", torch.allclose(out_f, suffix, atol=1e-5),
          f"max err {(out_f - suffix).abs().max():.2e}")
    check("B = global mean", torch.allclose(out_b, glob, atol=1e-5),
          f"max err {(out_b - glob).abs().max():.2e}")
    check("C,F = mean of the two halves",
          torch.allclose(out_cf, (prefix + suffix) / 2, atol=1e-5),
          f"max err {(out_cf - (prefix + suffix) / 2).abs().max():.2e}")

    print("\n[A2] masks block information in the right direction")
    attn_c = FullWidthMaskedAttention(d, "C", dropout=0.0).eval()
    attn_f = FullWidthMaskedAttention(d, "F", dropout=0.0).eval()
    x_last = x.clone()
    x_last[:, -1, :] += 100.0
    x_first = x.clone()
    x_first[:, 0, :] += 100.0
    with torch.no_grad():
        c_a, c_b = attn_c(x), attn_c(x_last)
        f_a, f_b = attn_f(x), attn_f(x_first)
    check("causal head cannot see the last position",
          torch.allclose(c_a[:, :-1], c_b[:, :-1], atol=1e-5),
          f"max err {(c_a[:, :-1] - c_b[:, :-1]).abs().max():.2e}")
    check("future head cannot see the first position",
          torch.allclose(f_a[:, 1:], f_b[:, 1:], atol=1e-5),
          f"max err {(f_a[:, 1:] - f_b[:, 1:]).abs().max():.2e}")

    print("\n[A3] padding and degenerate rows")
    pad = torch.zeros(B, T, dtype=torch.bool)
    pad[:, -2:] = True
    with torch.no_grad():
        out = uniform_attention("B", d)(x, padding_mask=pad)
    check("padded keys excluded from the average",
          torch.allclose(out[:, :-2], x[:, :-2].mean(dim=1, keepdim=True), atol=1e-5))
    with torch.no_grad():
        out = uniform_attention("C", d)(x, padding_mask=torch.ones(B, T, dtype=torch.bool))
    check("fully masked rows produce no NaN", bool(torch.isfinite(out).all()))

    # Regression test. A future-only head at a padded query position sees nothing but
    # padding, so its whole row is masked. Blocking with -inf makes softmax return NaN
    # there; patching the forward value afterwards still leaves a NaN *gradient*, which
    # wipes out every parameter on the first step. The forward check above passes either
    # way -- only the backward catches it.
    attn = FullWidthMaskedAttention(d, "C,F", dropout=0.0)
    x_grad = x.clone().requires_grad_(True)
    attn(x_grad, padding_mask=pad).sum().backward()
    nonfinite = [
        n for n, p in attn.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    check("gradients stay finite when a row is fully masked", not nonfinite,
          f"non-finite in: {nonfinite}" if nonfinite else "")

    print("\n[A4] the recovery map")
    attn = FullWidthMaskedAttention(d, "C,F")
    check("out_proj maps H*d -> d", tuple(attn.out_proj.weight.shape) == (d, 2 * d),
          str(tuple(attn.out_proj.weight.shape)))
    check("out_proj is a pure linear map (no bias)", attn.out_proj.bias is None)
    check("each head keeps full width d",
          all(m.weight.shape == (d, d) for m in attn.v_proj))
    check("'C,F' and 'CF' parse alike",
          parse_mask_spec("C,F") == parse_mask_spec("CF") == ("C", "F"))


def build_iter(task, split, max_tokens):
    task.load_dataset(split)
    return task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        ignore_invalid_inputs=True,
        num_workers=0,
    ).next_epoch_itr(shuffle=False)


def part_b(args):
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    state = checkpoint_utils.load_checkpoint_to_cpu(args.teacher_checkpoint)
    cfg = state["cfg"]
    cfg.task.data = args.data
    task = tasks.setup_task(cfg.task)

    print(f"\nteacher : {args.teacher_checkpoint}")
    print(f"arch    : {getattr(cfg.model, '_name', '?')}")
    print(f"updates : {state.get('optimizer_history', [{}])[-1].get('num_updates', -1)}")

    student_args = Namespace(
        arch="simple_attn_student_base",
        teacher_checkpoint=args.teacher_checkpoint,
        head_mask_spec=args.head_mask_spec,
        student_embed=args.student_embed,
        dropout=0.0,
    )
    student = models.build_model(student_args, task).to(device)
    student.train()

    print("\n[B1] model wiring")
    check("teacher is a simple_attn_teacher",
          isinstance(student.teacher, SimpleAttnTeacherModel))
    check("teacher is unmasked (spec B)",
          student.teacher.encoder.self_attn.mask_spec == ("B",),
          str(student.teacher.encoder.self_attn.mask_spec))
    check("student carries the requested spec",
          student.encoder.self_attn.mask_spec == parse_mask_spec(args.head_mask_spec),
          str(student.encoder.self_attn.mask_spec))
    check("student has no FFN (it regresses the attention output)",
          student.encoder.attn_layer_norm is None)

    shared = (
        student.encoder.embed_tokens.weight
        is student.teacher.encoder.embed_tokens.weight
    )
    if args.student_embed == "shared":
        check("student reuses the teacher's embedding tensor", shared)
    else:
        check("student embedding is its own tensor, not the teacher's", not shared)
        if args.student_embed == "copy":
            check("student embedding initialised from the teacher's",
                  torch.equal(student.encoder.embed_tokens.weight.detach(),
                              student.teacher.encoder.embed_tokens.weight.detach()))

    print("\n[B2] teacher frozen")
    unfrozen = [n for n, p in student.teacher.named_parameters() if p.requires_grad]
    check("no teacher parameter requires grad", not unfrozen,
          f"{len(unfrozen)} unfrozen" if unfrozen else "")
    check("teacher stays in eval mode after model.train()",
          not student.teacher.training)
    frozen_student = [
        n for n, p in student.named_parameters()
        if not n.startswith("teacher.") and not p.requires_grad
    ]
    n_trainable = sum(
        p.numel() for n, p in student.named_parameters()
        if not n.startswith("teacher.") and p.requires_grad
    )
    if args.student_embed == "shared":
        check("only the shared embedding is frozen on the student side",
              all("embed_tokens" in n for n in frozen_student),
              f"frozen: {frozen_student}")
    else:
        check("every student parameter is trainable", not frozen_student,
              f"frozen: {frozen_student}" if frozen_student else f"{n_trainable/1e6:.1f}M params")

    print("\n[B3] regression optimises, teacher unchanged")
    itr = build_iter(task, args.split, args.max_tokens)
    batch = next(iter(itr))
    net_input = {
        k: v.to(device) for k, v in batch["net_input"].items() if torch.is_tensor(v)
    }

    before = {n: p.detach().clone() for n, p in student.teacher.named_parameters()}
    embed_before = student.encoder.embed_tokens.weight.detach().clone()
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)

    losses = []
    nan_step = None
    for step in range(args.steps):
        out, extra = student(**net_input)
        target = extra["teacher_attn_out"]
        pad = extra["padding_mask"]
        if pad is not None:
            out, target = out[~pad], target[~pad]
        loss = (out.float() - target.float()).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if nan_step is None and not math.isfinite(losses[-1]):
            nan_step = step

    check("student and teacher outputs have the same shape",
          out.shape == target.shape, f"{tuple(out.shape)}")
    check(f"loss decreased over {args.steps} steps",
          nan_step is None and losses[-1] < losses[0],
          f"{losses[0]:.5f} -> {losses[-1]:.5f}"
          + (f", first non-finite at step {nan_step}" if nan_step is not None else ""))
    drifted = [
        n for n, p in student.teacher.named_parameters()
        if not torch.equal(p.detach(), before[n])
    ]
    check("teacher weights bit-identical after student updates", not drifted,
          f"drifted: {drifted[:3]}" if drifted else "")
    if args.student_embed != "shared":
        # The student's table must move: rows for tokens in the batch get gradient,
        # and it must not be the teacher's tensor (checked above), so the teacher
        # staying identical while this changes is the pair of facts that matter.
        check("student embedding table received updates",
              not torch.equal(
                  student.encoder.embed_tokens.weight.detach(), embed_before))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-checkpoint", default="",
                   help="if given, also run part B against this trained teacher")
    p.add_argument("--data", default="", help="binarised data dir (part B)")
    p.add_argument("--split", default="valid")
    p.add_argument("--head-mask-spec", default="C,F")
    p.add_argument("--student-embed", default="random",
                   choices=("random", "copy", "shared"),
                   help="must match what the training run uses")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    print("=" * 62)
    print("  Part A -- attention and mask behaviour (no data needed)")
    print("=" * 62)
    part_a()

    if args.teacher_checkpoint:
        if not args.data:
            print("\n--data is required alongside --teacher-checkpoint", file=sys.stderr)
            return 2
        print("\n" + "=" * 62)
        print("  Part B -- frozen teacher and student optimisation")
        print("=" * 62)
        part_b(args)
    else:
        print("\n(part B skipped: pass --teacher-checkpoint and --data to run it)")

    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 62)
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
        return 1
    print("  PASS -- the recovery setup behaves as intended.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
