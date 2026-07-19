#!/usr/bin/env python3
"""Run cross-condition generalization experiments for WICV-Net.

For every protocol directory produced by scripts/build_cross_condition_splits.py,
train the requested variants (default: ce_only baseline and full WICV-Net) and
evaluate on the held-out domain. The headline claim this table supports:
WICV-Net degrades less than the baseline when the test domain is unseen.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from run_ablation import VARIANTS

PROTOCOLS = ["norain2rain", "rain2norain", "morning2evening", "evening2morning"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-root", default="/mnt/recover/ngan/vehicles/reid_cross_condition")
    parser.add_argument("--protocols", nargs="+", default=PROTOCOLS, choices=PROTOCOLS)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["ce_only", "full"],
        choices=list(VARIANTS.keys()),
    )
    parser.add_argument("--model-name", default="osnet_x1_0")
    parser.add_argument("--results-root", default="results/wicv_cross_condition")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w-adv", type=float, default=None, help="Override adversarial weight for all runs.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    protocol_root = Path(args.protocol_root)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for protocol in args.protocols:
        split_root = protocol_root / protocol
        if not (split_root / "train.csv").exists():
            print(f"Missing protocol splits: {split_root} (run scripts/build_cross_condition_splits.py first)", file=sys.stderr)
            return 1

        for variant in args.variants:
            output_dir = results_root / f"{protocol}_{args.model_name}_{variant}"
            eval_path = output_dir / "eval.json"
            if args.skip_existing and eval_path.exists():
                print(f"Skipping {protocol}/{variant}: {eval_path} exists", flush=True)
            else:
                train_command = [
                    sys.executable, "-u", str(script_dir / "train.py"),
                    "--train-csv", str(split_root / "train.csv"),
                    "--val-query", str(split_root / "val_query.csv"),
                    "--val-gallery", str(split_root / "val_gallery.csv"),
                    "--model-name", args.model_name,
                    "--output-dir", str(output_dir),
                    "--epochs", str(args.epochs),
                    "--eval-every", str(args.eval_every),
                    "--patience", str(args.patience),
                    "--num-workers", str(args.num_workers),
                    "--seed", str(args.seed),
                    *VARIANTS[variant],
                ]
                if args.w_adv is not None:
                    train_command += ["--w-adv", str(args.w_adv)]
                run(train_command)

                checkpoint = output_dir / "model_best.pth"
                if not checkpoint.exists():
                    checkpoint = output_dir / "model_last.pth"
                run([
                    sys.executable, "-u", str(script_dir / "evaluate.py"),
                    "--checkpoint", str(checkpoint),
                    "--query", str(split_root / "query.csv"),
                    "--gallery", str(split_root / "gallery.csv"),
                    "--num-workers", str(args.num_workers),
                    "--output", str(eval_path),
                ])

            result = json.loads(eval_path.read_text(encoding="utf-8"))
            summary_rows.append(
                {
                    "protocol": protocol,
                    "variant": variant,
                    "model_name": args.model_name,
                    "rank1": result["overall"]["rank1"],
                    "rank5": result["overall"]["rank5"],
                    "mAP": result["overall"]["mAP"],
                }
            )

    summary_path = results_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["protocol", "variant", "model_name", "rank1", "rank5", "mAP"])
        writer.writeheader()
        writer.writerows(summary_rows)
    (results_root / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")

    for row in summary_rows:
        print(
            f"{row['protocol']:>16} {row['variant']:>14}: rank1={row['rank1']:.4f} "
            f"rank5={row['rank5']:.4f} mAP={row['mAP']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
