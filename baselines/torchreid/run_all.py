#!/usr/bin/env python3
"""Run multiple Torchreid baselines and aggregate their evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = [
    "osnet_x1_0",
    "osnet_ain_x1_0",
    "osnet_ibn_x1_0",
    "resnet50",
    "resnet101",
    "mobilenetv2_x1_0",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="/mnt/ngan/vehicles/reid_crops/manifest.csv",
        help="Crop manifest used to auto-build identity splits if split CSV files are missing.",
    )
    parser.add_argument("--train-csv", default="/mnt/ngan/vehicles/reid_benchmark_identity/train.csv")
    parser.add_argument("--val-query", default="/mnt/ngan/vehicles/reid_benchmark_identity/val_query.csv")
    parser.add_argument("--val-gallery", default="/mnt/ngan/vehicles/reid_benchmark_identity/val_gallery.csv")
    parser.add_argument("--query", default="/mnt/ngan/vehicles/reid_benchmark_identity/query.csv")
    parser.add_argument("--gallery", default="/mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv")
    parser.add_argument(
        "--split-output-root",
        default="",
        help="Output root for auto-built train/query/gallery split. Defaults to the train CSV parent.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-root", default="results/baselines")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Torchreid model names to train and evaluate.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and evaluate existing model_last.pth files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a model if its eval.json already exists.",
    )
    parser.add_argument(
        "--no-auto-split",
        action="store_true",
        help="Do not auto-build missing train/query/gallery split files.",
    )
    return parser.parse_args()


def safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def ensure_split_files(args: argparse.Namespace, results_root: Path) -> None:
    split_paths = [
        Path(args.train_csv),
        Path(args.val_query),
        Path(args.val_gallery),
        Path(args.query),
        Path(args.gallery),
    ]
    missing = [path for path in split_paths if not path.exists()]
    if not missing:
        return

    if args.no_auto_split:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing split files: {missing_text}. "
            "Run scripts/build_train_test_split.py first or remove --no-auto-split."
        )

    split_output_root = Path(args.split_output_root) if args.split_output_root else Path(args.train_csv).parent
    print("Missing identity split files; building them first.", flush=True)
    print(f"Missing: {', '.join(str(path) for path in missing)}", flush=True)
    command = [
        sys.executable,
        "-u",
        "scripts/build_train_test_split.py",
        "--manifest",
        args.manifest,
        "--output-root",
        str(split_output_root),
        "--train-ratio",
        str(args.train_ratio),
        "--val-ratio",
        str(args.val_ratio),
        "--seed",
        str(args.seed),
    ]
    run_command(command, results_root / "build_train_test_split.log")

    still_missing = [path for path in split_paths if not path.exists()]
    if still_missing:
        missing_text = ", ".join(str(path) for path in still_missing)
        raise FileNotFoundError(f"Split build finished but files are still missing: {missing_text}")


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(command)}", flush=True)
    print(f"Log: {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def train_command(args: argparse.Namespace, model_name: str, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "baselines/torchreid/train.py",
        "--train-csv",
        args.train_csv,
        "--val-query",
        args.val_query,
        "--val-gallery",
        args.val_gallery,
        "--model-name",
        model_name,
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(args.epochs),
        "--eval-every",
        str(args.eval_every),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
    ]
    if args.device:
        command.extend(["--device", args.device])
    return command


def eval_command(args: argparse.Namespace, model_name: str, output_dir: Path, eval_path: Path) -> list[str]:
    weights_path = output_dir / "model_best.pth"
    if not weights_path.exists():
        weights_path = output_dir / "model_last.pth"
    command = [
        sys.executable,
        "-u",
        "baselines/torchreid/evaluate.py",
        "--query",
        args.query,
        "--gallery",
        args.gallery,
        "--model-name",
        model_name,
        "--weights",
        str(weights_path),
        "--output",
        str(eval_path),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.device:
        command.extend(["--device", args.device])
    return command


def write_summary(results_root: Path, rows: list[dict]) -> None:
    json_path = results_root / "summary.json"
    csv_path = results_root / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fields = [
        "model",
        "rank1",
        "rank5",
        "mAP",
        "valid_queries",
        "num_query_images",
        "num_gallery_images",
        "weights",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})

    print(f"Summary JSON: {json_path}", flush=True)
    print(f"Summary CSV : {csv_path}", flush=True)


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    ensure_split_files(args, results_root)

    all_results = []
    for model_name in args.models:
        model_dir = results_root / safe_name(model_name)
        eval_path = model_dir / "eval.json"

        if args.skip_existing and eval_path.exists():
            print(f"Skipping existing result: {model_name}", flush=True)
        else:
            if not args.skip_train:
                run_command(
                    train_command(args, model_name, model_dir),
                    model_dir / "train.log",
                )
            run_command(
                eval_command(args, model_name, model_dir, eval_path),
                model_dir / "eval.log",
            )

        if eval_path.exists():
            result = json.loads(eval_path.read_text(encoding="utf-8"))
            result["model"] = model_name
            all_results.append(result)
            write_summary(results_root, all_results)

    print("\nFinal results:", flush=True)
    for result in all_results:
        print(
            f"{result['model']}: "
            f"Rank-1={result['rank1'] * 100:.2f}, "
            f"Rank-5={result['rank5'] * 100:.2f}, "
            f"mAP={result['mAP'] * 100:.2f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
