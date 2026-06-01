#!/usr/bin/env python3
"""Build identity-disjoint train/query/gallery splits for Re-ID training."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--manifest", default="/mnt/ngan/vehicles/reid_crops/manifest.csv")
    parser.add_argument("--output-root", default="/mnt/ngan/vehicles/reid_benchmark_identity")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
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
    return row["condition"], row["vehicle_id"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in FIELDS})


def print_stats(name: str, rows: list[dict]) -> None:
    ids = {key(row) for row in rows}
    by_condition = Counter(row["condition"] for row in rows)
    by_label = Counter(row["label"] for row in rows)
    print(f"{name}: images={len(rows)}, identities={len(ids)}")
    print(f"  by_condition={dict(sorted(by_condition.items()))}")
    print(f"  by_label={dict(sorted(by_label.items()))}")


def main() -> int:
    args = parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        print("--train-ratio must be between 0 and 1", file=sys.stderr)
        return 1

    manifest_path = Path(args.manifest)
    output_root = Path(args.output_root)
    rows = read_manifest(manifest_path)
    for row in rows:
        row["view_group"] = normalize_view(row["view"])

    before_keys = {key(row) for row in rows if row["view_group"] == "before"}
    after_keys = {key(row) for row in rows if row["view_group"] == "after"}
    shared_keys = before_keys & after_keys

    keys_by_condition: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for identity_key in sorted(shared_keys):
        keys_by_condition[identity_key[0]].append(identity_key)

    rng = random.Random(args.seed)
    train_keys = set()
    test_keys = set()
    for condition, condition_keys in sorted(keys_by_condition.items()):
        shuffled = condition_keys[:]
        rng.shuffle(shuffled)
        train_count = max(1, int(round(len(shuffled) * args.train_ratio)))
        train_count = min(train_count, len(shuffled) - 1)
        train_keys.update(shuffled[:train_count])
        test_keys.update(shuffled[train_count:])
        print(
            f"{condition}: shared_identities={len(shuffled)}, "
            f"train={train_count}, test={len(shuffled) - train_count}"
        )

    train = [row for row in rows if key(row) in train_keys]
    query = [
        row
        for row in rows
        if key(row) in test_keys and row["view_group"] == "after"
    ]
    gallery = [
        row
        for row in rows
        if key(row) in test_keys and row["view_group"] == "before"
    ]

    write_csv(output_root / "train.csv", train)
    write_csv(output_root / "query.csv", query)
    write_csv(output_root / "gallery.csv", gallery)

    print(f"Manifest: {manifest_path}")
    print(f"Output root: {output_root}")
    print(f"Shared identities: {len(shared_keys)}")
    print_stats("train", train)
    print_stats("query", query)
    print_stats("gallery", gallery)
    return 0


if __name__ == "__main__":
    sys.exit(main())
