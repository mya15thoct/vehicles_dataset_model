#!/usr/bin/env python3
"""Hyperparameter sensitivity study for WICV-Net loss weights.

Sweeps one parameter at a time around the default configuration. The w_adv
sweep doubles as the fix for the adversarial-weight issue observed in the
first ablation (no_adv outperforming full at w_adv=0.5): it locates the value
where condition invariance helps instead of hurting.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

SWEEPS: dict[str, list[float]] = {
    "w_adv": [0.0, 0.05, 0.1, 0.25, 0.5],
    "w_cvpa": [0.1, 0.25, 0.5, 1.0],
    "w_tri": [0.5, 1.0, 2.0],
    "temperature": [0.04, 0.07, 0.1],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/train.csv")
    parser.add_argument("--val-query", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/val_query.csv")
    parser.add_argument("--val-gallery", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/val_gallery.csv")
    parser.add_argument("--query", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/query.csv")
    parser.add_argument("--gallery", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/gallery.csv")
    parser.add_argument("--model-name", default="osnet_x1_0")
    parser.add_argument("--results-root", default="results/wicv_sensitivity")
    parser.add_argument("--params", nargs="+", default=["w_adv", "w_cvpa"], choices=list(SWEEPS.keys()))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
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
    for param in args.params:
        for value in SWEEPS[param]:
            run_name = f"{args.model_name}_{param}_{value}"
            output_dir = results_root / run_name
            eval_path = output_dir / "eval.json"
            if args.skip_existing and eval_path.exists():
                print(f"Skipping {run_name}: eval.json exists", flush=True)
            else:
                train_command = [
                    sys.executable, "-u", str(script_dir / "train.py"),
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
                    f"--{param.replace('_', '-')}", str(value),
                ]
                run(train_command)

                checkpoint = output_dir / "model_best.pth"
                if not checkpoint.exists():
                    checkpoint = output_dir / "model_last.pth"
                run([
                    sys.executable, "-u", str(script_dir / "evaluate.py"),
                    "--checkpoint", str(checkpoint),
                    "--query", args.query,
                    "--gallery", args.gallery,
                    "--num-workers", str(args.num_workers),
                    "--output", str(eval_path),
                ])

            result = json.loads(eval_path.read_text(encoding="utf-8"))
            best_val_path = output_dir / "best_val.json"
            best_val = json.loads(best_val_path.read_text(encoding="utf-8")) if best_val_path.exists() else {}
            summary_rows.append(
                {
                    "param": param,
                    "value": value,
                    "model_name": args.model_name,
                    "val_mAP": best_val.get("mAP"),
                    "rank1": result["overall"]["rank1"],
                    "rank5": result["overall"]["rank5"],
                    "mAP": result["overall"]["mAP"],
                }
            )

    summary_path = results_root / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["param", "value", "model_name", "val_mAP", "rank1", "rank5", "mAP"])
        writer.writeheader()
        writer.writerows(summary_rows)
    (results_root / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")

    for row in summary_rows:
        print(
            f"{row['param']}={row['value']}: val_mAP={row['val_mAP']} "
            f"test rank1={row['rank1']:.4f} mAP={row['mAP']:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
