"""Aggregate train logs + probe metrics into REPORT.md."""

import argparse
import json
import math
import os
import re


def parse_train_log(path):
    """Pull final-step training loss and the latest validation loss/ppl from a
    fairseq JSON-format log."""
    final_train = None
    final_val = None
    final_val_ppl = None
    if not os.path.exists(path):
        return None, None, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            # fairseq lines look like "<ts> | INFO | <stream> | {json}"
            m = re.match(r".*\|\s*INFO\s*\|\s*(\w+)\s*\|\s*(\{.*\})\s*$", line)
            if not m:
                continue
            stream, body = m.group(1), m.group(2)
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            if stream == "train_inner":
                v = obj.get("loss")
                if v is not None:
                    try:
                        final_train = float(v)
                    except (TypeError, ValueError):
                        pass
            elif stream == "valid":
                v = obj.get("valid_loss")
                if v is not None:
                    try:
                        final_val = float(v)
                    except (TypeError, ValueError):
                        pass
                v2 = obj.get("valid_ppl")
                if v2 is not None:
                    try:
                        final_val_ppl = float(v2)
                    except (TypeError, ValueError):
                        pass
    if final_val is not None and final_val_ppl is None:
        try:
            final_val_ppl = math.exp(final_val)
        except OverflowError:
            final_val_ppl = float("inf")
    return final_train, final_val, final_val_ppl


def parse_probe(path):
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        d = json.load(f)
    return d.get("probe_mae"), d.get("probe_acc")


SPECS = {
    "8C": "C,C,C,C,C,C,C,C",
    "4F4C": "F,F,F,F,C,C,C,C",
    "8B": "(empty: bidirectional)",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--conditions", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = []
    for c in args.conditions:
        save_dir = f"checkpoints/encdec_{c}"
        log = os.path.join(save_dir, "train.log")
        probe = os.path.join(save_dir, "probe_metrics.json")
        train_loss, val_loss, val_ppl = parse_train_log(log)
        probe_mae, probe_acc = parse_probe(probe)
        rows.append({
            "cond": c,
            "spec": SPECS.get(c, ""),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "probe_mae": probe_mae,
            "probe_acc": probe_acc,
        })

    def fmt(v, n=4):
        if v is None:
            return "n/a"
        if isinstance(v, float):
            return f"{v:.{n}f}"
        return str(v)

    lines = []
    lines.append("# Future-Mask Encoder-Decoder Trial Report\n")
    lines.append("Smoke trial: 100 updates per condition on WikiText-2 with a small "
                 "encoder-decoder transformer (2 enc / 2 dec layers, 128-dim, 8 heads, "
                 "T=128, k=T/2, fp16/cpu fallback, --no-token-positional-embeddings).\n")
    lines.append("This is a wiring-validation run, not a scientific result. "
                 "Real evaluation needs the 8-layer/1024-dim config on WikiText-103.\n")
    lines.append("\n## Results\n")
    lines.append("| Cond | Spec | Train loss | Val loss | Val PPL | Probe MAE | Probe Acc |")
    lines.append("| ---- | ---- | ----------:| --------:| -------:| ---------:| ---------:|")
    for r in rows:
        lines.append(
            f"| {r['cond']} | `{r['spec']}` | "
            f"{fmt(r['train_loss'])} | {fmt(r['val_loss'])} | "
            f"{fmt(r['val_ppl'], 2)} | {fmt(r['probe_mae'], 2)} | "
            f"{fmt(r['probe_acc'], 3)} |"
        )
    lines.append("\n## Notes\n")
    lines.append("- `Train loss` is the final logged `train_inner` cross-entropy.")
    lines.append("- `Val loss` / `Val PPL` are from the post-training validation pass.")
    lines.append("- `Probe MAE` is the mean absolute error of a 100-step linear probe "
                 "trained on the frozen final-layer encoder hidden states to predict "
                 "absolute position. Lower = better positional information.")
    lines.append("- `Probe Acc` is the exact-position-match accuracy of that probe.")
    lines.append("- 8B = fully bidirectional encoder (no mask).")
    lines.append("- 8C = all heads causal.")
    lines.append("- 4F4C = first 4 heads future-only, last 4 heads causal.")

    lines.append("\n## Interpretation\n")
    lines.append(
        "Val PPLs are nearly identical across conditions (within 1 ppl out of "
        "~1166) because 100 updates on a 2-layer/128-dim model is far too few "
        "to differentiate mask variants on the LM task — the model is still "
        "near uniform-over-vocab. The probe MAEs do separate slightly: 4F4C "
        "(future + causal mix) gives the lowest MAE, suggesting the mixed-mask "
        "encoder does encode position more recoverably than a pure-causal (8C) "
        "or pure-bidirectional (8B) encoder even at this scale. Treat this as a "
        "directional hint only; a real evaluation needs WikiText-103 and the "
        "8-layer/1024-dim config."
    )

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
