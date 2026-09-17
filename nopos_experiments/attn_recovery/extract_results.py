"""Recover student metrics from fairseq train logs.

fairseq emits its JSON records through the logging module, so a line reads

    2026-09-17 10:00:00 | INFO | valid | {"epoch": 1, "valid_loss": "0.02", ...}

-- the JSON is a *suffix*, not the whole line (``progress_bar.py:217``, with the
format set in ``fairseq_cli/train.py:19``). Anything matching on ``line.startswith("{")``
silently finds nothing. Validation records are identified here by their ``valid_``
prefixed keys rather than by anything in the log prefix, so the format of that prefix
does not matter.

Two modes:

  # one log -> one result file (what exp_attn_recovery.sh calls)
  python extract_results.py --log <train.log> --out <result.json> \
      --condition CF --spec C,F --seed 1924

  # recover everything from a finished or interrupted run
  python extract_results.py --scan-root checkpoints_attn_recovery \
      --results-dir attn_recovery_results --seed 1924
"""

import argparse
import json
import os
import sys

KEYS = ("valid_loss", "valid_mae", "valid_rel_l2", "valid_cos", "valid_mae_baseline")


def parse_log(path):
    """Return (metrics from the last validation record, number of JSON records seen)."""
    found = {}
    seen_json = 0
    with open(path) as f:
        for line in f:
            brace = line.find("{")
            if brace < 0:
                continue
            try:
                rec = json.loads(line[brace:])
            except ValueError:
                continue
            seen_json += 1
            if not any(k.startswith("valid_") for k in rec):
                continue
            for k in KEYS:
                if k in rec:
                    found[k] = float(rec[k])
            if "valid_num_updates" in rec:
                try:
                    found["num_updates"] = int(float(rec["valid_num_updates"]))
                except ValueError:
                    pass
    return found, seen_json


def result_dict(condition, spec, seed, found):
    return {
        "condition": condition,
        "head_mask_spec": spec,
        "seed": seed,
        "mae": found.get("valid_mae", float("nan")),
        "rel_l2": found.get("valid_rel_l2", float("nan")),
        "cos": found.get("valid_cos", float("nan")),
        "mae_baseline": found.get("valid_mae_baseline", float("nan")),
        "num_updates": found.get("num_updates", -1),
    }


def report_empty(path, seen_json):
    """Say what the log actually held, so a failure is diagnosable without a re-run."""
    print(f"      parsed {seen_json} JSON records, none with valid_* keys", file=sys.stderr)
    try:
        with open(path) as f:
            tail = [ln.rstrip() for ln in f if "valid" in ln][-3:]
    except OSError:
        tail = []
    for ln in tail:
        print("      | " + ln[:200], file=sys.stderr)


def write_one(log, out, condition, spec, seed):
    found, seen_json = parse_log(log)
    if not found:
        report_empty(log, seen_json)
        print(f"FATAL: no validation records found in {log}", file=sys.stderr)
        return False
    with open(out, "w") as f:
        json.dump(result_dict(condition, spec, seed, found), f, indent=2)
    print(f"      Saved {out}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", help="a single train.log to parse")
    p.add_argument("--out", help="result JSON to write (with --log)")
    p.add_argument("--condition", default="")
    p.add_argument("--spec", default="")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--scan-root", help="checkpoints root to walk for <COND>_seed<SEED>/train.log")
    p.add_argument("--results-dir", help="where to write result JSONs (with --scan-root)")
    p.add_argument("--overwrite", action="store_true",
                   help="rewrite result files that already exist")
    args = p.parse_args()

    if args.log:
        if not args.out:
            p.error("--out is required with --log")
        spec = args.spec or ",".join(args.condition)
        return 0 if write_one(args.log, args.out, args.condition, spec, args.seed) else 1

    if not args.scan_root or not args.results_dir:
        p.error("pass either --log/--out or --scan-root/--results-dir")

    os.makedirs(args.results_dir, exist_ok=True)
    suffix = f"_seed{args.seed}"
    dirs = sorted(
        d for d in os.listdir(args.scan_root)
        if d.endswith(suffix) and os.path.isdir(os.path.join(args.scan_root, d))
    )
    if not dirs:
        print(f"no */{suffix} directories under {args.scan_root}", file=sys.stderr)
        return 1

    wrote = skipped = failed = 0
    for d in dirs:
        condition = d[: -len(suffix)]
        if condition.startswith("teacher"):
            continue
        log = os.path.join(args.scan_root, d, "train.log")
        out = os.path.join(args.results_dir, f"{d}.json")
        if not os.path.exists(log):
            print(f"      {d}: no train.log -- skipping")
            skipped += 1
            continue
        if os.path.exists(out) and not args.overwrite:
            print(f"      {d}: result exists -- skipping (use --overwrite)")
            skipped += 1
            continue
        if write_one(log, out, condition, ",".join(condition), args.seed):
            wrote += 1
        else:
            failed += 1

    print(f"\n  wrote {wrote}, skipped {skipped}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
