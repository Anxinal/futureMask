"""Train an MLP probe on the element-wise product of Q and K vectors from
a frozen encoder's self-attention to predict the signed relative distance
between two tokens. Reports MAE, accuracy, and directional accuracy on
the validation split.

The probe input is  q_i ⊙ k_j  (element-wise product of the query at
position i and the key at position j), which is the per-dimension
decomposition of the attention logit  q_i · k_j.

Usage:
  python run_relative_position_probe.py \
      --checkpoint checkpoints/encdec_8C/checkpoint_last.pt \
      --data data-bin/wikitext-103 \
      --split valid \
      --probe-updates 5000 \
      --probe-layer -1 \
      --output rel_probe_metrics.json
"""

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import checkpoint_utils, tasks, utils


def _build_iter(task, split, max_tokens):
    task.load_dataset(split)
    dataset = task.dataset(split)
    batch_iter = task.get_batch_iterator(
        dataset=dataset,
        max_tokens=max_tokens,
        ignore_invalid_inputs=True,
        num_workers=0,
    ).next_epoch_itr(shuffle=False)
    return batch_iter


def _encoder_qk(model, sample, layer_idx, device):
    """Extract Q and K vectors from a specific encoder layer's self-attention.

    Returns:
        q: [B, T, C] query vectors
        k: [B, T, C] key vectors
        not_pad: [B, T] boolean mask of non-pad positions
    """
    src_tokens = sample["net_input"]["src_tokens"].to(device)
    src_lengths = sample["net_input"]["src_lengths"].to(device)
    with torch.no_grad():
        encoder_out = model.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=True,
        )
    # encoder_states: [embedding_out, layer0_out, layer1_out, ...]
    # Input to layer L = encoder_states[L]
    states = encoder_out["encoder_states"]

    num_layers = len(model.encoder.layers)
    lid = layer_idx if layer_idx >= 0 else num_layers + layer_idx
    layer = model.encoder.layers[lid]

    x = states[lid]  # [T, B, C] — input to this layer

    # Pre-norm: layer norm is applied before Q/K projections
    if getattr(layer, "normalize_before", False):
        x = layer.self_attn_layer_norm(x)

    q = layer.self_attn.q_proj(x)  # [T, B, C]
    k = layer.self_attn.k_proj(x)  # [T, B, C]

    q = q.transpose(0, 1).contiguous()  # [B, T, C]
    k = k.transpose(0, 1).contiguous()  # [B, T, C]

    pad_idx = _task.source_dictionary.pad()
    not_pad = src_tokens.ne(pad_idx)
    return q, k, not_pad


def _sample_pairs(not_pad, pairs_per_seq, device):
    """Sample random token pairs from non-pad positions in each sequence.

    Returns:
        batch_idx [N]: which sequence each pair belongs to
        pos_i [N]: position of first token in each pair
        pos_j [N]: position of second token in each pair
        distances [N]: signed relative distance (pos_j - pos_i)
    """
    B, T = not_pad.shape
    all_batch, all_i, all_j, all_dist = [], [], [], []

    for b in range(B):
        valid = not_pad[b].nonzero(as_tuple=False).squeeze(-1)  # [V]
        V = valid.size(0)
        if V < 2:
            continue

        K = min(pairs_per_seq, V * (V - 1))

        # Sample random index pairs into valid positions
        idx_i = torch.randint(V, (K,), device=device)
        idx_j = torch.randint(V, (K,), device=device)

        # Reject i == j collisions by resampling
        collisions = idx_i == idx_j
        while collisions.any():
            idx_j[collisions] = torch.randint(V, (collisions.sum(),), device=device)
            collisions = idx_i == idx_j

        pi = valid[idx_i]
        pj = valid[idx_j]

        all_batch.append(torch.full((K,), b, device=device, dtype=torch.long))
        all_i.append(pi)
        all_j.append(pj)
        all_dist.append(pj - pi)  # signed distance

    if not all_batch:
        empty = torch.zeros(0, device=device, dtype=torch.long)
        return empty, empty, empty, empty

    return (
        torch.cat(all_batch),
        torch.cat(all_i),
        torch.cat(all_j),
        torch.cat(all_dist),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="valid")
    p.add_argument("--train-split", default="train")
    p.add_argument("--probe-updates", type=int, default=5000)
    p.add_argument("--probe-layer", type=int, default=-1)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--num-rel-classes", type=int, default=1024)
    p.add_argument("--pairs-per-seq", type=int, default=64)
    p.add_argument("--eval-tokens-per-sample", type=int, default=0,
                   help="override tokens_per_sample at eval time (0 = use training value)")
    p.add_argument("--output", required=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    num_classes = args.num_rel_classes
    offset = num_classes // 2  # class 512 = distance 0

    state = checkpoint_utils.load_checkpoint_to_cpu(args.checkpoint)
    cfg = state["cfg"]
    cfg.task.data = args.data

    # Override sequence length for extrapolation experiments
    if args.eval_tokens_per_sample > 0:
        cfg.task.tokens_per_sample = args.eval_tokens_per_sample
        cfg.model.max_source_positions = args.eval_tokens_per_sample
        cfg.model.max_target_positions = args.eval_tokens_per_sample

    global _task
    _task = tasks.setup_task(cfg.task)
    model = _task.build_model(cfg.model)
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    embed_dim = model.encoder.embed_tokens.embedding_dim
    probe = nn.Sequential(
        nn.Linear(embed_dim, num_classes),
        nn.ReLU(),
        nn.Linear(num_classes, num_classes),
    ).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)

    # ---- Train ----
    train_iter = _build_iter(_task, args.train_split, args.max_tokens)
    train_batches = list(train_iter)
    if not train_batches:
        raise RuntimeError("no training batches for probe")
    print(f"rel_probe: {len(train_batches)} train batches available", flush=True)

    step = 0
    while step < args.probe_updates:
        for sample in train_batches:
            if step >= args.probe_updates:
                break

            q, k, not_pad = _encoder_qk(model, sample, args.probe_layer, device)
            batch_idx, pos_i, pos_j, distances = _sample_pairs(
                not_pad, args.pairs_per_seq, device
            )
            if batch_idx.numel() == 0:
                continue

            qk = q[batch_idx, pos_i] * k[batch_idx, pos_j]  # [N, C]
            logits = probe(qk)  # [N, num_classes]
            labels = (distances + offset).clamp(0, num_classes - 1)
            loss = F.cross_entropy(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                print(f"rel_probe step {step} loss {loss.item():.4f}", flush=True)

    # ---- Eval ----
    probe.eval()
    val_iter = _build_iter(_task, args.split, args.max_tokens)
    abs_errs = []
    correct = 0
    dir_correct = 0
    total = 0

    with torch.no_grad():
        for sample in val_iter:
            q, k, not_pad = _encoder_qk(model, sample, args.probe_layer, device)
            batch_idx, pos_i, pos_j, distances = _sample_pairs(
                not_pad, args.pairs_per_seq, device
            )
            if batch_idx.numel() == 0:
                continue

            qk = q[batch_idx, pos_i] * k[batch_idx, pos_j]
            logits = probe(qk)
            labels = (distances + offset).clamp(0, num_classes - 1)
            preds = logits.argmax(dim=-1)

            # Convert back to signed distances for metrics
            pred_dist = preds - offset
            true_dist = distances

            abs_errs.append((pred_dist - true_dist).abs().float().cpu().numpy())
            correct += (preds == labels).sum().item()

            # Directional accuracy: does the probe get the sign right?
            # Exclude distance-0 pairs from directional metric
            nonzero = true_dist != 0
            if nonzero.any():
                dir_correct += (
                    (pred_dist[nonzero].sign() == true_dist[nonzero].sign()).sum().item()
                )
            total += labels.numel()

    mae = float(np.concatenate(abs_errs).mean()) if abs_errs else float("nan")
    acc = correct / max(1, total)
    dir_acc = dir_correct / max(1, total)
    print(
        f"rel_probe: MAE={mae:.3f} acc={acc:.3f} dir_acc={dir_acc:.3f} n={total}",
        flush=True,
    )
    with open(args.output, "w") as f:
        json.dump(
            {
                "rel_probe_mae": mae,
                "rel_probe_acc": acc,
                "rel_probe_dir_acc": dir_acc,
                "rel_probe_n": total,
            },
            f,
        )


if __name__ == "__main__":
    sys.exit(main())
