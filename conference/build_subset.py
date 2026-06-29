#!/usr/bin/env python3
"""Build a condition-balanced conference subset for Re-ID experiments."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


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
    parser.add_argument("--manifest", default="/mnt/ngan/vehicles/reid_crops_full/manifest.csv")
    parser.add_argument("--output-root", default="/mnt/ngan/vehicles/reid_benchmark_conference_50")
    parser.add_argument("--ids-per-condition", type=int, default=300)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=[],
        help="Optional condition names to include. Defaults to all conditions found in the manifest.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def normalize_view(view: str) -> str:
    if view.startswith("before"):
        return "before"
    if view.startswith("after"):
        return "after"
    return view


def key(row: dict) -> tuple[str, str]:
    return row["condition"], str(int(row["vehicle_id"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in FIELDS})


def write_selected_identities(path: Path, split_keys: dict[str, set[tuple[str, str]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["condition", "vehicle_id", "split"])
        writer.writeheader()
        for split_name, keys in split_keys.items():
            for condition, vehicle_id in sorted(keys):
                writer.writerow(
                    {
                        "condition": condition,
                        "vehicle_id": vehicle_id,
                        "split": split_name,
                    }
                )


def summarize_rows(rows: list[dict]) -> dict:
    identities = {key(row) for row in rows}
    crop_paths = {row["crop_path"] for row in rows}
    return {
        "rows": len(rows),
        "identities": len(identities),
        "duplicate_crop_paths": len(rows) - len(crop_paths),
        "by_condition": dict(sorted(Counter(row["condition"] for row in rows).items())),
        "by_view": dict(sorted(Counter(normalize_view(row["view"]) for row in rows).items())),
        "by_label": dict(sorted(Counter(row["label"] for row in rows).items())),
    }


def print_stats(name: str, rows: list[dict]) -> None:
    summary = summarize_rows(rows)
    print(f"{name}: images={summary['rows']}, identities={summary['identities']}")
    print(f"  by_condition={summary['by_condition']}")
    print(f"  by_view={summary['by_view']}")
    print(f"  by_label={summary['by_label']}")


def validate_args(args: argparse.Namespace) -> int:
    if args.ids_per_condition <= 0:
        print("--ids-per-condition must be positive", file=sys.stderr)
        return 1
    if not 0.0 < args.train_ratio < 1.0:
        print("--train-ratio must be between 0 and 1", file=sys.stderr)
        return 1
    if not 0.0 <= args.val_ratio < 1.0:
        print("--val-ratio must be between 0 and 1", file=sys.stderr)
        return 1
    if args.train_ratio + args.val_ratio >= 1.0:
        print("--train-ratio + --val-ratio must be less than 1", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    invalid = validate_args(args)
    if invalid:
        return invalid

    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)
    rows = read_manifest(manifest_path)
    for row in rows:
        row["view_group"] = normalize_view(row["view"])

    requested_conditions = set(args.conditions)
    if requested_conditions:
        rows = [row for row in rows if row["condition"] in requested_conditions]

    before_keys = {key(row) for row in rows if row["view_group"] == "before"}
    after_keys = {key(row) for row in rows if row["view_group"] == "after"}
    shared_keys = before_keys & after_keys

    keys_by_condition: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for identity_key in sorted(shared_keys):
        keys_by_condition[identity_key[0]].append(identity_key)

    if requested_conditions:
        missing_conditions = requested_conditions - set(keys_by_condition)
        if missing_conditions:
            print(f"Conditions not found in shared IDs: {sorted(missing_conditions)}", file=sys.stderr)
            return 1

    rng = random.Random(args.seed)
    split_keys = {"train": set(), "val": set(), "test": set()}
    selected_by_condition: dict[str, dict] = {}

    for condition, condition_keys in sorted(keys_by_condition.items()):
        available = len(condition_keys)
        if available < args.ids_per_condition:
            print(
                f"{condition}: only {available} shared identities, "
                f"but --ids-per-condition={args.ids_per_condition}",
                file=sys.stderr,
            )
            return 1

        shuffled = condition_keys[:]
        rng.shuffle(shuffled)
        selected = shuffled[: args.ids_per_condition]
        count = len(selected)
        train_count = int(round(count * args.train_ratio))
        val_count = int(round(count * args.val_ratio))
        train_count = min(train_count, count - val_count - 1)
        if train_count < 1:
            train_count = 1
        if train_count + val_count >= count:
            val_count = max(0, count - train_count - 1)
        test_count = count - train_count - val_count

        train = set(selected[:train_count])
        val = set(selected[train_count : train_count + val_count])
        test = set(selected[train_count + val_count :])
        split_keys["train"].update(train)
        split_keys["val"].update(val)
        split_keys["test"].update(test)
        selected_by_condition[condition] = {
            "available_shared_ids": available,
            "selected_shared_ids": count,
            "train_ids": train_count,
            "val_ids": val_count,
            "test_ids": test_count,
        }
        print(
            f"{condition}: available={available}, selected={count}, "
            f"train={train_count}, val={val_count}, test={test_count}"
        )

    selected_keys = split_keys["train"] | split_keys["val"] | split_keys["test"]
    train_rows = [row for row in rows if key(row) in split_keys["train"]]
    val_query = [
        row
        for row in rows
        if key(row) in split_keys["val"] and row["view_group"] == "after"
    ]
    val_gallery = [
        row
        for row in rows
        if key(row) in split_keys["val"] and row["view_group"] == "before"
    ]
    query = [
        row
        for row in rows
        if key(row) in split_keys["test"] and row["view_group"] == "after"
    ]
    gallery = [
        row
        for row in rows
        if key(row) in split_keys["test"] and row["view_group"] == "before"
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    write_csv(output_root / "train.csv", train_rows)
    write_csv(output_root / "val_query.csv", val_query)
    write_csv(output_root / "val_gallery.csv", val_gallery)
    write_csv(output_root / "query.csv", query)
    write_csv(output_root / "gallery.csv", gallery)
    write_selected_identities(output_root / "selected_identities.csv", split_keys)

    stats = {
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "seed": args.seed,
        "ids_per_condition": args.ids_per_condition,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "total_available_shared_ids": len(shared_keys),
        "total_selected_shared_ids": len(selected_keys),
        "selected_by_condition": selected_by_condition,
        "splits": {
            "train": summarize_rows(train_rows),
            "val_query": summarize_rows(val_query),
            "val_gallery": summarize_rows(val_gallery),
            "query": summarize_rows(query),
            "gallery": summarize_rows(gallery),
        },
    }
    (output_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Manifest: {manifest_path}")
    print(f"Output root: {output_root}")
    print(f"Selected shared identities: {len(selected_keys)}")
    print_stats("train", train_rows)
    print_stats("val_query", val_query)
    print_stats("val_gallery", val_gallery)
    print_stats("query", query)
    print_stats("gallery", gallery)
    print(f"Saved: {output_root / 'selected_identities.csv'}")
    print(f"Saved: {output_root / 'stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
