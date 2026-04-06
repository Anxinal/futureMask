#!/usr/bin/env python3
"""Run the WikiText-103 mixed future-mask NoPos experiment and summarize losses."""

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT_DIR = REPO_ROOT / "examples" / "language_model" / "wikitext-103"
DEFAULT_DATA_BIN = REPO_ROOT / "data-bin" / "wikitext-103"
DEFAULT_SAVE_ROOT = REPO_ROOT / "nopos_experiments" / "future_mask_wiki103" / "runs"
DEFAULT_VENDOR_DIR = REPO_ROOT / ".vendor_cpuexp"
DEFAULT_HEAD_COUNTS = [0, 1, 2, 4]


@dataclass
class RunResult:
    future_mask_heads: int
    save_dir: str
    train_loss: Optional[float]
    last_valid_loss: Optional[float]
    best_valid_loss: Optional[float]
    best_valid_num_updates: Optional[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--text-dir", default=str(DEFAULT_TEXT_DIR))
    parser.add_argument("--data-bin", default=str(DEFAULT_DATA_BIN))
    parser.add_argument("--save-root", default=str(DEFAULT_SAVE_ROOT))
    parser.add_argument("--vendor-dir", default=str(DEFAULT_VENDOR_DIR))
    parser.add_argument(
        "--future-head-counts",
        default="0,1,2,4",
        help="comma-separated counts of future-masked heads to train",
    )
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-update", type=int, default=286000)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--t-mult", type=int, default=2)
    parser.add_argument("--lr-period-updates", type=int, default=270000)
    parser.add_argument("--lr-shrink", type=float, default=0.75)
    parser.add_argument("--warmup-updates", type=int, default=16000)
    parser.add_argument("--warmup-init-lr", type=float, default=1e-7)
    parser.add_argument("--stop-min-lr", type=float, default=1e-9)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--clip-norm", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--update-freq", type=int, default=3)
    parser.add_argument("--tokens-per-sample", type=int, default=3072)
    parser.add_argument("--validate-interval-updates", type=int, default=1000)
    parser.add_argument("--keep-best-checkpoints", type=int, default=1)
    parser.add_argument("--arch", default="transformer_lm_wiki103")
    parser.add_argument("--criterion", default="adaptive_loss")
    parser.add_argument("--ddp-backend", default="legacy_ddp")
    parser.add_argument("--optimizer", default="nag")
    parser.add_argument("--sample-break-mode", default="none")
    parser.add_argument("--required-batch-size-multiple", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-user-site", action="store_true")
    return parser.parse_args()


def parse_head_counts(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def ensure_dataset_files(text_dir: Path) -> None:
    required = [
        text_dir / "wiki.train.tokens",
        text_dir / "wiki.valid.tokens",
        text_dir / "wiki.test.tokens",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing WikiText-103 source files:\n{}".format("\n".join(missing))
        )


def run_command(cmd: List[str], dry_run: bool, env: Dict[str, str]) -> None:
    print(" ".join(subprocess.list2cmdline([part]) for part in cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def preprocess_if_needed(args: argparse.Namespace, env: Dict[str, str]) -> None:
    if args.skip_preprocess or args.summarize_only:
        return

    text_dir = Path(args.text_dir)
    data_bin = Path(args.data_bin)
    ensure_dataset_files(text_dir)
    if (data_bin / "dict.txt").exists() and not args.force:
        print(f"Skipping preprocess because {(data_bin / 'dict.txt')} already exists.")
        return

    data_bin.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        "-m",
        "fairseq_cli.preprocess",
        "--only-source",
        "--trainpref",
        str(text_dir / "wiki.train.tokens"),
        "--validpref",
        str(text_dir / "wiki.valid.tokens"),
        "--testpref",
        str(text_dir / "wiki.test.tokens"),
        "--destdir",
        str(data_bin),
        "--workers",
        str(args.workers),
    ]
    run_command(cmd, dry_run=args.dry_run, env=env)


def build_train_command(
    args: argparse.Namespace,
    future_heads: int,
    save_dir: Path,
    log_file: Path,
) -> List[str]:
    cmd = [
        args.python,
        "-m",
        "fairseq_cli.train",
        "--task",
        "language_modeling",
        str(Path(args.data_bin)),
        "--save-dir",
        str(save_dir),
        "--arch",
        args.arch,
        "--decoder-attention-heads",
        "8",
        "--no-token-positional-embeddings",
        "--future-mask-heads",
        str(future_heads),
        "--criterion",
        args.criterion,
        "--max-update",
        str(args.max_update),
        "--lr",
        str(args.lr),
        "--t-mult",
        str(args.t_mult),
        "--lr-period-updates",
        str(args.lr_period_updates),
        "--lr-scheduler",
        "cosine",
        "--lr-shrink",
        str(args.lr_shrink),
        "--warmup-updates",
        str(args.warmup_updates),
        "--warmup-init-lr",
        str(args.warmup_init_lr),
        "--stop-min-lr",
        str(args.stop_min_lr),
        "--optimizer",
        args.optimizer,
        "--min-lr",
        str(args.min_lr),
        "--clip-norm",
        str(args.clip_norm),
        "--max-tokens",
        str(args.max_tokens),
        "--update-freq",
        str(args.update_freq),
        "--tokens-per-sample",
        str(args.tokens_per_sample),
        "--seed",
        str(args.seed),
        "--sample-break-mode",
        args.sample_break_mode,
        "--skip-invalid-size-inputs-valid-test",
        "--ddp-backend",
        args.ddp_backend,
        "--required-batch-size-multiple",
        str(args.required_batch_size_multiple),
        "--num-workers",
        str(args.num_workers),
        "--validate-interval-updates",
        str(args.validate_interval_updates),
        "--keep-best-checkpoints",
        str(args.keep_best_checkpoints),
        "--log-format",
        "json",
        "--log-interval",
        str(args.log_interval),
        "--log-file",
        str(log_file),
    ]
    if args.cpu:
        cmd.append("--cpu")
    if args.fp16:
        cmd.append("--fp16")
    if args.no_save:
        cmd.append("--no-save")
    return cmd


def extract_json_payload(line: str) -> Optional[dict]:
    line = line.strip()
    if "{" not in line:
        return None
    payload = line[line.index("{") :]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def parse_train_log(log_file: Path, future_heads: int, save_dir: Path) -> RunResult:
    train_loss = None
    last_valid_loss = None
    best_valid_loss = None
    best_valid_num_updates = None

    if not log_file.exists():
        return RunResult(
            future_mask_heads=future_heads,
            save_dir=str(save_dir),
            train_loss=None,
            last_valid_loss=None,
            best_valid_loss=None,
            best_valid_num_updates=None,
        )

    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        payload = extract_json_payload(line)
        if payload is None:
            continue
        if "train_loss" in payload:
            train_loss = float(payload["train_loss"])
        if "valid_loss" in payload:
            current_valid = float(payload["valid_loss"])
            last_valid_loss = current_valid
            if best_valid_loss is None or current_valid < best_valid_loss:
                best_valid_loss = current_valid
                best_valid_num_updates = int(
                    payload.get("valid_num_updates", payload.get("num_updates", 0))
                )

    return RunResult(
        future_mask_heads=future_heads,
        save_dir=str(save_dir),
        train_loss=train_loss,
        last_valid_loss=last_valid_loss,
        best_valid_loss=best_valid_loss,
        best_valid_num_updates=best_valid_num_updates,
    )


def write_summary_files(results: List[RunResult], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"
    md_path = output_root / "report.md"

    json_path.write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    md_path.write_text(build_markdown_report(results), encoding="utf-8")


def build_markdown_report(results: List[RunResult]) -> str:
    lines = [
        "# WikiText-103 Mixed Future-Mask Experiment",
        "",
        "| Future-mask heads | Train loss | Last valid loss | Best valid loss | Best valid updates |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]

    for result in results:
        lines.append(
            "| {future_mask_heads} | {train_loss} | {last_valid_loss} | "
            "{best_valid_loss} | {best_valid_num_updates} |".format(
                future_mask_heads=result.future_mask_heads,
                train_loss=format_metric(result.train_loss),
                last_valid_loss=format_metric(result.last_valid_loss),
                best_valid_loss=format_metric(result.best_valid_loss),
                best_valid_num_updates=result.best_valid_num_updates or "n/a",
            )
        )

    lines.extend(["", "## Conclusion", "", build_conclusion(results), ""])
    return "\n".join(lines)


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def build_conclusion(results: List[RunResult]) -> str:
    baseline = next((result for result in results if result.future_mask_heads == 0), None)
    comparable = [result for result in results if result.best_valid_loss is not None]

    if not comparable:
        return (
            "No training logs were available to summarize. Run the experiment first "
            "and regenerate the report."
        )

    best = min(comparable, key=lambda result: result.best_valid_loss)
    if baseline is None or baseline.best_valid_loss is None:
        return (
            "The report is missing a comparable 0-head causal baseline, so no "
            "baseline-relative conclusion can be drawn automatically."
        )

    delta = best.best_valid_loss - baseline.best_valid_loss
    if best.future_mask_heads == 0:
        return (
            "The pure causal NoPos baseline achieved the best validation loss in "
            "this run. Adding future-masked heads did not improve validation loss."
        )

    if delta < 0:
        return (
            f"The best mixed-mask setting used {best.future_mask_heads} future-masked "
            f"heads and improved best validation loss by {-delta:.4f} relative to "
            "the pure causal NoPos baseline."
        )

    return (
        f"The best mixed-mask setting used {best.future_mask_heads} future-masked "
        f"heads, but it was worse than the pure causal NoPos baseline by {delta:.4f} "
        "validation-loss points."
    )


def main() -> None:
    args = parse_args()
    future_head_counts = parse_head_counts(args.future_head_counts)
    save_root = Path(args.save_root)
    env = dict(os.environ)
    python_paths = [str(REPO_ROOT)]
    vendor_dir = Path(args.vendor_dir)
    if vendor_dir.exists():
        python_paths.insert(0, str(vendor_dir))
    existing_pythonpath = env.get("PYTHONPATH", "")
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    if args.no_user_site:
        env["PYTHONNOUSERSITE"] = "1"

    preprocess_if_needed(args, env)

    results: List[RunResult] = []
    for future_heads in future_head_counts:
        run_name = f"future_heads_{future_heads}"
        save_dir = save_root / run_name
        log_file = save_dir / "train.log"
        save_dir.mkdir(parents=True, exist_ok=True)

        should_run = not args.summarize_only and (
            args.force or not (save_dir / "checkpoint_best.pt").exists()
        )
        if should_run:
            cmd = build_train_command(args, future_heads, save_dir, log_file)
            run_command(cmd, dry_run=args.dry_run, env=env)
        else:
            print(f"Skipping training for {run_name}; existing checkpoint found.")

        results.append(parse_train_log(log_file, future_heads, save_dir))

    if results:
        write_summary_files(results, save_root)
        print(f"Wrote summary files under {save_root}")


if __name__ == "__main__":
    main()
