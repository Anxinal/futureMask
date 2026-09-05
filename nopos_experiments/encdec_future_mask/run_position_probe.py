"""Train a linear probe on a frozen encoder's hidden states to predict
absolute token position. Reports MAE + accuracy on the validation split.

Usage:
  python run_position_probe.py \
      --checkpoint checkpoints/encdec_8C/checkpoint_last.pt \
      --data data-bin/wikitext-2 \
      --split valid \
      --probe-updates 100 \
      --probe-layer -1 \
      --output probe_metrics.json
"""

import argparse
import json
import os
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


def _encoder_hidden(model, sample, layer_idx, device):
    src_tokens = sample["net_input"]["src_tokens"].to(device)
    src_lengths = sample["net_input"]["src_lengths"].to(device)
    with torch.no_grad():
        encoder_out = model.encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=True,
        )
    states = encoder_out["encoder_states"]  # list of [T, B, C]
    h = states[layer_idx]
    h = h.transpose(0, 1).contiguous()  # [B, T, C]
    pad_idx = task.source_dictionary.pad()
    not_pad = src_tokens.ne(pad_idx)
    return h, not_pad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="valid")
    p.add_argument("--train-split", default="train")
    p.add_argument("--probe-updates", type=int, default=100)
    p.add_argument("--probe-layer", type=int, default=-1)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--eval-tokens-per-sample", type=int, default=0,
                   help="override tokens_per_sample at eval time (0 = use training value)")
    p.add_argument("--output", required=True)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    state = checkpoint_utils.load_checkpoint_to_cpu(args.checkpoint)
    cfg = state["cfg"]
    # Point the task at the (possibly different) local data dir.
    cfg.task.data = args.data

    # Override sequence length for extrapolation experiments
    if args.eval_tokens_per_sample > 0:
        cfg.task.tokens_per_sample = args.eval_tokens_per_sample
        cfg.model.max_source_positions = args.eval_tokens_per_sample
        cfg.model.max_target_positions = args.eval_tokens_per_sample

    global task  # used by _encoder_hidden
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    embed_dim = model.encoder.embed_tokens.embedding_dim
    max_pos = int(getattr(cfg.task, "tokens_per_sample", 256)) + 4
    probe = nn.Sequential(
        nn.Linear(embed_dim, max_pos),
        nn.ReLU(),
        nn.Linear(max_pos, max_pos),
    ).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)

    pad_idx = task.source_dictionary.pad()

    train_iter = _build_iter(task, args.train_split, args.max_tokens)
    train_batches = list(train_iter)
    if not train_batches:
        raise RuntimeError("no training batches for probe")
    print(f"probe: {len(train_batches)} train batches available", flush=True)

    step = 0
    while step < args.probe_updates:
        for sample in train_batches:
            if step >= args.probe_updates:
                break
            h, not_pad = _encoder_hidden(model, sample, args.probe_layer, device)
            B, T, _ = h.shape
            positions = (
                torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            )
            logits = probe(h)
            loss = F.cross_entropy(
                logits[not_pad], positions[not_pad], reduction="mean"
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 10 == 0:
                print(f"probe step {step} loss {loss.item():.4f}", flush=True)

    # Eval.
    probe.eval()
    val_iter = _build_iter(task, args.split, args.max_tokens)
    abs_errs = []
    correct = 0
    total = 0
    with torch.no_grad():
        for sample in val_iter:
            h, not_pad = _encoder_hidden(model, sample, args.probe_layer, device)
            B, T, _ = h.shape
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            logits = probe(h)
            preds = logits.argmax(dim=-1)
            mask = not_pad
            abs_errs.append((preds[mask] - positions[mask]).abs().float().cpu().numpy())
            correct += (preds[mask] == positions[mask]).sum().item()
            total += int(mask.sum().item())

    mae = float(np.concatenate(abs_errs).mean()) if abs_errs else float("nan")
    acc = correct / max(1, total)
    print(f"probe: MAE={mae:.3f} acc={acc:.3f} n={total}", flush=True)
    with open(args.output, "w") as f:
        json.dump({"probe_mae": mae, "probe_acc": acc, "probe_n": total}, f)


if __name__ == "__main__":
    sys.exit(main())
