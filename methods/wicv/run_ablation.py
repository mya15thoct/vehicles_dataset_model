#!/usr/bin/env python3
"""Run the WICV-Net ablation study: full model plus one-component-removed variants.

Variants (all sharing the same backbone, sampler, schedule, and seed):
  ce_only        -- identity cross-entropy only (framework lower bound)
  plain_triplet  -- CE + standard batch-hard triplet (strong generic baseline)
  no_adv         -- full model without the condition-adversarial heads
  no_cvpa        -- full model without cross-view prototype alignment
  no_cvtri       -- full model with plain instead of cross-view triplet mining
  full           -- complete WICV-Net objective

Each variant is trained, then its best checkpoint is evaluated on the test
split (overall + per condition). Results are aggregated into summary.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

VARIANTS: dict[str, list[str]] = {
    "ce_only": ["--no-triplet", "--no-adv", "--no-cvpa"],
    "plain_triplet": ["--plain-triplet", "--no-adv", "--no-cvpa"],
    "no_adv": ["--no-adv"],
    "no_cvpa": ["--no-cvpa"],
    "no_cvtri": ["--plain-triplet"],
    "full": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="/mnt/ngan/vehicles/reid_benchmark_identity/train.csv")
    parser.add_argument("--val-query", default="/mnt/ngan/vehicles/reid_benchmark_identity/val_query.csv")
    parser.add_argument("--val-gallery", default="/mnt/ngan/vehicles/reid_benchmark_identity/val_gallery.csv")
    parser.add_argument("--query", default="/mnt/ngan/vehicles/reid_benchmark_identity/query.csv")
    parser.add_argument("--gallery", default="/mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv")
    parser.add_argument("--model-name", default="osnet_x1_0")
    parser.add_argument("--results-root", default="results/wicv_ablation")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VARIANTS.keys()),
        choices=list(VARIANTS.keys()),
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip variants that already have eval.json.")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for variant in args.variants:
        output_dir = results_root / f"{args.model_name}_{variant}"
        eval_path = output_dir / "eval.json"
        if args.skip_existing and eval_path.exists():
            print(f"Skipping {variant}: {eval_path} exists", flush=True)
        else:
            train_command = [
                sys.executable,
                "-u",
                str(script_dir / "train.py"),
                "--train-csv", args.train_csv,
                "--val-query", args.val_query,
                "--val-gallery", args.val_gallery,
                "--model-name", args.model_name,
                "--output-dir", str(output_dir),
                "--epochs", str(args.epochs),
                "--eval-every", str(args.eval_every),
                "--patience", str(args.patience),
                "--num-workers", str(args.num_workers),
                "--seed", str(args.seed),
                *VARIANTS[variant],
            ]
            run(train_command)

            checkpoint = output_dir / "model_best.pth"
            if not checkpoint.exists():
                checkpoint = output_dir / "model_last.pth"
            eval_command = [
                sys.executable,
                "-u",
                str(script_dir / "evaluate.py"),
                "--checkpoint", str(checkpoint),
                "--query", args.query,
                "--gallery", args.gallery,
                "--num-workers", str(args.num_workers),
                "--output", str(eval_path),
            ]
            run(eval_command)

        result = json.loads(eval_path.read_text(encoding="utf-8"))
        row = {
            "variant": variant,
            "model_name": args.model_name,
            "rank1": result["overall"]["rank1"],
            "rank5": result["overall"]["rank5"],
            "mAP": result["overall"]["mAP"],
        }
        for condition, metrics in sorted(result.get("per_condition", {}).items()):
            row[f"{condition}_rank1"] = metrics["rank1"]
            row[f"{condition}_mAP"] = metrics["mAP"]
        summary_rows.append(row)

    fieldnames = sorted({field for row in summary_rows for field in row}, key=lambda name: (name not in ("variant", "model_name", "rank1", "rank5", "mAP"), name))
    summary_path = results_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    (results_root / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")

    for row in summary_rows:
        print(
            f"{row['variant']:>14}: rank1={row['rank1']:.4f} "
            f"rank5={row['rank5']:.4f} mAP={row['mAP']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
