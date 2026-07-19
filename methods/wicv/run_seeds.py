#!/usr/bin/env python3
"""Train WICV-Net with multiple random seeds and report mean +/- std.

Produces the statistical-robustness numbers for the main results table
(e.g. "84.9 +/- 0.4"), which preempts the reviewer question of whether the
improvement is within run-to-run noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

from run_ablation import VARIANTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/train.csv")
    parser.add_argument("--val-query", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/val_query.csv")
    parser.add_argument("--val-gallery", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/val_gallery.csv")
    parser.add_argument("--query", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/query.csv")
    parser.add_argument("--gallery", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full/gallery.csv")
    parser.add_argument("--model-name", default="osnet_x1_0")
    parser.add_argument("--variant", default="full", choices=list(VARIANTS.keys()))
    parser.add_argument("--results-root", default="results/wicv_seeds")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--w-adv", type=float, default=None, help="Override adversarial weight.")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in args.seeds:
        output_dir = results_root / f"{args.model_name}_{args.variant}_seed{seed}"
        eval_path = output_dir / "eval.json"
        if args.skip_existing and eval_path.exists():
            print(f"Skipping seed {seed}: eval.json exists", flush=True)
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
                "--seed", str(seed),
                *VARIANTS[args.variant],
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
                "--query", args.query,
                "--gallery", args.gallery,
                "--num-workers", str(args.num_workers),
                "--output", str(eval_path),
            ])

        result = json.loads(eval_path.read_text(encoding="utf-8"))
        row = {
            "seed": seed,
            "rank1": result["overall"]["rank1"],
            "rank5": result["overall"]["rank5"],
            "mAP": result["overall"]["mAP"],
        }
        for condition, metrics in sorted(result.get("per_condition", {}).items()):
            row[f"{condition}_mAP"] = metrics["mAP"]
        per_seed.append(row)

    metric_names = [name for name in per_seed[0] if name != "seed"]
    aggregate = {}
    for name in metric_names:
        mean, std = mean_std([row[name] for row in per_seed])
        aggregate[name] = {"mean": mean, "std": std}

    summary = {
        "model_name": args.model_name,
        "variant": args.variant,
        "seeds": args.seeds,
        "w_adv_override": args.w_adv,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    summary_path = results_root / f"{args.model_name}_{args.variant}_seeds_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = results_root / f"{args.model_name}_{args.variant}_seeds_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["seed"] + metric_names)
        writer.writeheader()
        writer.writerows(per_seed)

    print(f"Saved: {summary_path}")
    for name in ("rank1", "rank5", "mAP"):
        stats = aggregate[name]
        print(f"{name}: {stats['mean']:.4f} +/- {stats['std']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
