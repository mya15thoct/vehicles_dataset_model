#!/usr/bin/env python3
"""Audit Re-ID split CSV files for identity and crop leakage."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="/mnt/ngan/vehicles/reid_benchmark_identity/train.csv")
    parser.add_argument("--val-query", default="/mnt/ngan/vehicles/reid_benchmark_identity/val_query.csv")
    parser.add_argument("--val-gallery", default="/mnt/ngan/vehicles/reid_benchmark_identity/val_gallery.csv")
    parser.add_argument("--query", default="/mnt/ngan/vehicles/reid_benchmark_identity/query.csv")
    parser.add_argument("--gallery", default="/mnt/ngan/vehicles/reid_benchmark_identity/gallery.csv")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def identity(row: dict) -> str:
    return f"{row['condition']}::{int(row['vehicle_id']):06d}"


def view_group(row: dict) -> str:
    view = row["view"]
    if view.startswith("before"):
        return "before"
    if view.startswith("after"):
        return "after"
    return view


def id_set(rows: list[dict]) -> set[str]:
    return {identity(row) for row in rows}


def crop_set(rows: list[dict]) -> set[str]:
    return {row["crop_path"] for row in rows}


def summarize(name: str, rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "identities": len(id_set(rows)),
        "crop_paths": len(crop_set(rows)),
        "duplicate_crop_paths": len(rows) - len(crop_set(rows)),
        "by_condition": dict(sorted(Counter(row["condition"] for row in rows).items())),
        "by_view": dict(sorted(Counter(row["view"] for row in rows).items())),
        "by_view_group": dict(sorted(Counter(view_group(row) for row in rows).items())),
        "by_label": dict(sorted(Counter(row["label"] for row in rows).items())),
    }


def overlap(left: set[str], right: set[str]) -> int:
    return len(left & right)


def sample_overlap(left: set[str], right: set[str], limit: int = 10) -> list[str]:
    return sorted(left & right)[:limit]


def main() -> int:
    args = parse_args()
    splits = {
        "train": read_csv(Path(args.train)),
        "val_query": read_csv(Path(args.val_query)),
        "val_gallery": read_csv(Path(args.val_gallery)),
        "query": read_csv(Path(args.query)),
        "gallery": read_csv(Path(args.gallery)),
    }

    val = splits["val_query"] + splits["val_gallery"]
    test = splits["query"] + splits["gallery"]

    identity_sets = {
        "train": id_set(splits["train"]),
        "val": id_set(val),
        "test": id_set(test),
        "val_query": id_set(splits["val_query"]),
        "val_gallery": id_set(splits["val_gallery"]),
        "query": id_set(splits["query"]),
        "gallery": id_set(splits["gallery"]),
    }
    crop_sets = {
        "train": crop_set(splits["train"]),
        "val": crop_set(val),
        "test": crop_set(test),
        "val_query": crop_set(splits["val_query"]),
        "val_gallery": crop_set(splits["val_gallery"]),
        "query": crop_set(splits["query"]),
        "gallery": crop_set(splits["gallery"]),
    }

    checks = {
        "identity_overlap_train_val": overlap(identity_sets["train"], identity_sets["val"]),
        "identity_overlap_train_test": overlap(identity_sets["train"], identity_sets["test"]),
        "identity_overlap_val_test": overlap(identity_sets["val"], identity_sets["test"]),
        "crop_overlap_train_val": overlap(crop_sets["train"], crop_sets["val"]),
        "crop_overlap_train_test": overlap(crop_sets["train"], crop_sets["test"]),
        "crop_overlap_val_test": overlap(crop_sets["val"], crop_sets["test"]),
        "crop_overlap_query_gallery": overlap(crop_sets["query"], crop_sets["gallery"]),
        "crop_overlap_val_query_val_gallery": overlap(crop_sets["val_query"], crop_sets["val_gallery"]),
        "val_identity_missing_gallery": len(identity_sets["val_query"] - identity_sets["val_gallery"]),
        "val_identity_missing_query": len(identity_sets["val_gallery"] - identity_sets["val_query"]),
        "test_identity_missing_gallery": len(identity_sets["query"] - identity_sets["gallery"]),
        "test_identity_missing_query": len(identity_sets["gallery"] - identity_sets["query"]),
        "query_non_after_rows": sum(1 for row in splits["query"] if view_group(row) != "after"),
        "gallery_non_before_rows": sum(1 for row in splits["gallery"] if view_group(row) != "before"),
        "val_query_non_after_rows": sum(1 for row in splits["val_query"] if view_group(row) != "after"),
        "val_gallery_non_before_rows": sum(1 for row in splits["val_gallery"] if view_group(row) != "before"),
    }

    leak_checks = [
        "identity_overlap_train_val",
        "identity_overlap_train_test",
        "identity_overlap_val_test",
        "crop_overlap_train_val",
        "crop_overlap_train_test",
        "crop_overlap_val_test",
        "crop_overlap_query_gallery",
        "crop_overlap_val_query_val_gallery",
        "val_identity_missing_gallery",
        "val_identity_missing_query",
        "test_identity_missing_gallery",
        "test_identity_missing_query",
        "query_non_after_rows",
        "gallery_non_before_rows",
        "val_query_non_after_rows",
        "val_gallery_non_before_rows",
    ]
    failed = {name: checks[name] for name in leak_checks if checks[name] != 0}

    report = {
        "splits": {name: summarize(name, rows) for name, rows in splits.items()},
        "combined": {
            "val": summarize("val", val),
            "test": summarize("test", test),
        },
        "checks": checks,
        "samples": {
            "identity_overlap_train_test": sample_overlap(identity_sets["train"], identity_sets["test"]),
            "identity_overlap_train_val": sample_overlap(identity_sets["train"], identity_sets["val"]),
            "identity_overlap_val_test": sample_overlap(identity_sets["val"], identity_sets["test"]),
            "crop_overlap_train_test": sample_overlap(crop_sets["train"], crop_sets["test"]),
            "crop_overlap_query_gallery": sample_overlap(crop_sets["query"], crop_sets["gallery"]),
        },
        "passed": not failed,
        "failed_checks": failed,
    }

    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved: {output_path}", flush=True)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
