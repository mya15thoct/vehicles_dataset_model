#!/usr/bin/env python3
"""Evaluate trained Re-ID checkpoints by condition and vehicle class."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CONDITIONS = [
    "morning_norain",
    "evening_norain",
    "morning_rain",
    "evening_rain",
]

DEFAULT_LABELS = [
    "bus",
    "car",
    "motorbike",
    "truck",
]

FIELDS = [
    "condition",
    "view",
    "vehicle_id",
    "label",
    "frame_id",
    "frame_name",
    "crop_path",
    "source_image",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Test query CSV.")
    parser.add_argument("--gallery", required=True, help="Test gallery CSV.")
    parser.add_argument(
        "--results-root",
        default="results/conference_50_e100",
        help="Directory containing trained model subdirectories.",
    )
    parser.add_argument(
        "--output-root",
        default="results/conference_50_breakdowns",
        help="Directory for filtered CSVs, logs, JSON metrics, and summaries.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["osnet_ain_x1_0"],
        help="Model names to evaluate. Use the best overall model for paper breakdown tables.",
    )
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def identity(row: dict) -> str:
    return f"{row['condition']}::{int(row['vehicle_id']):06d}"


def filter_pair(query_rows: list[dict], gallery_rows: list[dict], field: str, value: str) -> tuple[list[dict], list[dict]]:
    query_subset = [row for row in query_rows if row[field] == value]
    gallery_subset = [row for row in gallery_rows if row[field] == value]
    shared_ids = {identity(row) for row in query_subset} & {identity(row) for row in gallery_subset}
    query_subset = [row for row in query_subset if identity(row) in shared_ids]
    gallery_subset = [row for row in gallery_subset if identity(row) in shared_ids]
    return query_subset, gallery_subset


def weights_for(results_root: Path, model_name: str) -> Path:
    model_dir = results_root / model_name
    best = model_dir / "model_best.pth"
    if best.exists():
        return best
    last = model_dir / "model_last.pth"
    if last.exists():
        return last
    raise FileNotFoundError(f"Missing checkpoint for {model_name}: expected {best} or {last}")


def run_eval(
    model_name: str,
    weights: Path,
    query_csv: Path,
    gallery_csv: Path,
    output_json: Path,
    log_path: Path,
    args: argparse.Namespace,
) -> None:
    command = [
        sys.executable,
        "-u",
        "baselines/torchreid/evaluate.py",
        "--query",
        str(query_csv),
        "--gallery",
        str(gallery_csv),
        "--model-name",
        model_name,
        "--weights",
        str(weights),
        "--output",
        str(output_json),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.device:
        command.extend(["--device", args.device])

    output_json.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running: {' '.join(command)}", flush=True)
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


def summarize_rows(rows: list[dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "breakdown_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = [
        "breakdown",
        "subset",
        "model",
        "rank1",
        "rank5",
        "mAP",
        "valid_queries",
        "num_query_images",
        "num_gallery_images",
        "query_identities",
        "gallery_identities",
        "weights",
    ]
    with (output_root / "breakdown_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    query_rows = read_csv(Path(args.query))
    gallery_rows = read_csv(Path(args.gallery))
    results_root = Path(args.results_root)
    output_root = Path(args.output_root)
    subsets_root = output_root / "subsets"

    tasks = []
    for condition in args.conditions:
        q_rows, g_rows = filter_pair(query_rows, gallery_rows, "condition", condition)
        tasks.append(("condition", condition, q_rows, g_rows))
    for label in args.labels:
        q_rows, g_rows = filter_pair(query_rows, gallery_rows, "label", label)
        tasks.append(("class", label, q_rows, g_rows))

    summary = []
    for breakdown, subset, q_rows, g_rows in tasks:
        if not q_rows or not g_rows:
            print(f"Skipping empty subset: {breakdown}={subset}", flush=True)
            continue

        subset_dir = subsets_root / breakdown / subset
        query_csv = subset_dir / "query.csv"
        gallery_csv = subset_dir / "gallery.csv"
        write_csv(query_csv, q_rows)
        write_csv(gallery_csv, g_rows)

        q_ids = {identity(row) for row in q_rows}
        g_ids = {identity(row) for row in g_rows}
        print(
            f"{breakdown}={subset}: query={len(q_rows)} gallery={len(g_rows)} "
            f"shared_ids={len(q_ids & g_ids)}",
            flush=True,
        )

        for model_name in args.models:
            weights = weights_for(results_root, model_name)
            model_output_dir = output_root / model_name
            output_json = model_output_dir / f"{breakdown}_{subset}.json"
            log_path = model_output_dir / f"{breakdown}_{subset}.log"
            if args.skip_existing and output_json.exists():
                print(f"Skipping existing: {output_json}", flush=True)
            else:
                run_eval(model_name, weights, query_csv, gallery_csv, output_json, log_path, args)

            result = json.loads(output_json.read_text(encoding="utf-8"))
            result.update(
                {
                    "breakdown": breakdown,
                    "subset": subset,
                    "query_identities": len(q_ids),
                    "gallery_identities": len(g_ids),
                }
            )
            summary.append(result)
            summarize_rows(summary, output_root)

    summarize_rows(summary, output_root)
    print(f"Summary CSV : {output_root / 'breakdown_summary.csv'}", flush=True)
    print(f"Summary JSON: {output_root / 'breakdown_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
