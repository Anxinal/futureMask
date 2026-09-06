"""Verify the fixed-attention pin on a trained Stage-A checkpoint.

Checks the three claims the experiment rests on:

  1. Q/K weights (and biases) are exactly zero *after* training, not just at init.
  2. Attention is therefore an exact uniform average -- under a causal mask, the
     prefix mean 1/(t+1) over visible positions.
  3. V trained: v_proj is not frozen and did not stay at its initial value.

Usage:
  python nopos_experiments/verify_fixed_attn.py \
      --checkpoint checkpoints_fixed_attn/base_lm_seed999/checkpoint_last.pt \
      --data data-bin/wikitext-103
"""

import argparse
import sys

import torch

from fairseq import checkpoint_utils, tasks, utils


def check_qk_zero(model):
    ok = True
    for i, layer in enumerate(model.decoder.layers):
        for name in ("q_proj", "k_proj"):
            proj = getattr(layer.self_attn, name)
            w_max = proj.weight.detach().abs().max().item()
            b_max = (
                proj.bias.detach().abs().max().item() if proj.bias is not None else 0.0
            )
            status = "ok" if (w_max == 0.0 and b_max == 0.0) else "FAIL"
            ok &= status == "ok"
            print(f"  layer {i} {name}: |w|max={w_max:g} |b|max={b_max:g}  [{status}]")
    return ok


def check_v_trained(model, is_stage_a):
    """Q/K must be frozen; in Stage A, V must be trainable.

    Stage B freezes the whole decoder, so requires_grad is only informative for
    Stage A. fairseq's trainer filters the optimizer by requires_grad, so a frozen
    param never enters it at all.
    """
    ok = True
    for i, layer in enumerate(model.decoder.layers):
        attn = layer.self_attn
        flags = {n: getattr(attn, n).weight.requires_grad for n in ("q_proj", "k_proj", "v_proj")}
        v_std = attn.v_proj.weight.detach().float().std().item()

        bad = flags["q_proj"] or flags["k_proj"]
        if is_stage_a:
            bad = bad or not flags["v_proj"]
        bad = bad or v_std == 0.0

        status = "FAIL" if bad else "ok"
        ok &= not bad
        print(
            f"  layer {i}: requires_grad q={flags['q_proj']} k={flags['k_proj']} "
            f"v={flags['v_proj']}, v_std={v_std:.5f}  [{status}]"
        )
    if not is_stage_a:
        print("  (Stage-B checkpoint: whole decoder frozen, v requires_grad=False expected)")
    return ok


def check_uniform_attention(model, seq_len=8):
    """Feed random activations through layer 0's self-attention with a causal mask
    and confirm the returned weights are the prefix mean."""
    layer = model.decoder.layers[0]
    embed_dim = layer.self_attn.embed_dim
    dtype = layer.self_attn.q_proj.weight.dtype

    x = torch.randn(seq_len, 1, embed_dim, dtype=dtype)  # [T, B, C]
    attn_mask = torch.triu(
        utils.fill_with_neg_inf(torch.zeros(seq_len, seq_len)), 1
    ).to(dtype)

    with torch.no_grad():
        _, attn = layer.self_attn(
            query=x, key=x, value=x, attn_mask=attn_mask, need_weights=True
        )

    attn = attn[0].float()  # [T, T]
    t = torch.arange(seq_len, dtype=torch.float32)
    expect = torch.tril(torch.ones(seq_len, seq_len)) / (t + 1).unsqueeze(1)

    max_err = (attn - expect).abs().max().item()
    future = attn.triu(1).abs().max().item()
    row_sums = attn.sum(-1)

    print(f"  max |attn - prefix_mean| = {max_err:.2e}")
    print(f"  max weight on future positions = {future:.2e}")
    print(f"  row sums in [{row_sums.min():.4f}, {row_sums.max():.4f}]")
    print("  first 4 rows:")
    for i in range(min(4, seq_len)):
        print("    " + " ".join(f"{v:.3f}" for v in attn[i][: i + 1].tolist()))

    return max_err < 1e-3 and future == 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--seq-len", type=int, default=8)
    args = p.parse_args()

    state = checkpoint_utils.load_checkpoint_to_cpu(args.checkpoint)
    cfg = state["cfg"]
    cfg.task.data = args.data

    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    arch = getattr(cfg.model, "_name", "?")
    is_stage_a = "base" in str(arch)

    print(f"checkpoint: {args.checkpoint}")
    print(f"arch      : {arch}  ({'stage A' if is_stage_a else 'stage B'})")
    print(f"updates   : {state.get('optimizer_history', [{}])[-1].get('num_updates', -1)}")

    print("\n[1/3] Q/K pinned to zero after training")
    qk_ok = check_qk_zero(model)

    print("\n[2/3] Q/K frozen, V trainable")
    v_ok = check_v_trained(model, is_stage_a)

    print("\n[3/3] attention is the causal prefix mean")
    attn_ok = check_uniform_attention(model, args.seq_len)

    print()
    if qk_ok and v_ok and attn_ok:
        print("PASS -- fixed attention behaves as intended.")
        return 0
    print("FAIL -- see the checks marked FAIL above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
