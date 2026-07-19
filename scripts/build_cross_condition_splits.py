#!/usr/bin/env python3
"""Build cross-condition generalization protocols from the identity split.

Each protocol trains on conditions of one domain factor value and tests on the
held-out value, e.g. train on no-rain conditions and test on rain conditions.
Because every identity is scoped to a single condition, the held-out test pool
can safely include ALL identities of the held-out conditions (train, val, and
test identities of the main split alike) -- none of them were seen in
training. Validation stays inside the training domain so model selection never
peeks at the held-out domain.

Protocols:
  norain2rain      train morning_norain+evening_norain, test *_rain
  rain2norain      reverse
  morning2evening  train morning_*, test evening_*
  evening2morning  reverse

Output layout: <output-root>/<protocol>/{train,val_query,val_gallery,query,gallery}.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
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

# protocol -> (factor position in condition name, train value, test value)
PROTOCOLS = {
    "norain2rain": (1, "norain", "rain"),
    "rain2norain": (1, "rain", "norain"),
    "morning2evening": (0, "morning", "evening"),
    "evening2morning": (0, "evening", "morning"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full")
    parser.add_argument("--output-root", default="/mnt/recover/ngan/vehicles/reid_cross_condition")
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=list(PROTOCOLS.keys()),
        choices=list(PROTOCOLS.keys()),
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in FIELDS})


def factor_value(condition: str, position: int) -> str:
    return condition.split("_")[position]


def normalize_view(view: str) -> str:
    if view.startswith("before"):
        return "before"
    if view.startswith("after"):
        return "after"
    return view


def identity(row: dict) -> str:
    return f"{row['condition']}::{int(row['vehicle_id']):06d}"


def print_stats(name: str, rows: list[dict]) -> None:
    ids = {identity(row) for row in rows}
    by_condition = Counter(row["condition"] for row in rows)
    print(f"  {name}: images={len(rows)}, identities={len(ids)}, by_condition={dict(sorted(by_condition.items()))}")


def main() -> int:
    args = parse_args()
    split_root = Path(args.split_root)
    output_root = Path(args.output_root)

    files = {}
    for name in ("train", "val_query", "val_gallery", "query", "gallery"):
        path = split_root / f"{name}.csv"
        if not path.exists():
            print(f"Missing split file: {path}", file=sys.stderr)
            return 1
        files[name] = read_csv_rows(path)

    all_rows = [row for rows in files.values() for row in rows]

    for protocol in args.protocols:
        position, train_value, test_value = PROTOCOLS[protocol]
        print(f"== {protocol} (train={train_value}, test={test_value}) ==")

        train = [row for row in files["train"] if factor_value(row["condition"], position) == train_value]
        val_query = [row for row in files["val_query"] if factor_value(row["condition"], position) == train_value]
        val_gallery = [row for row in files["val_gallery"] if factor_value(row["condition"], position) == train_value]

        held_out = [row for row in all_rows if factor_value(row["condition"], position) == test_value]
        query = [row for row in held_out if normalize_view(row["view"]) == "after"]
        gallery = [row for row in held_out if normalize_view(row["view"]) == "before"]

        train_ids = {identity(row) for row in train}
        test_ids = {identity(row) for row in query} | {identity(row) for row in gallery}
        overlap = train_ids & test_ids
        if overlap:
            print(f"FATAL: {len(overlap)} identities overlap between train and test", file=sys.stderr)
            return 1
        if not train or not val_query or not val_gallery or not query or not gallery:
            print("FATAL: one of the splits is empty", file=sys.stderr)
            return 1

        protocol_root = output_root / protocol
        write_csv_rows(protocol_root / "train.csv", train)
        write_csv_rows(protocol_root / "val_query.csv", val_query)
        write_csv_rows(protocol_root / "val_gallery.csv", val_gallery)
        write_csv_rows(protocol_root / "query.csv", query)
        write_csv_rows(protocol_root / "gallery.csv", gallery)

        print_stats("train", train)
        print_stats("val_query", val_query)
        print_stats("val_gallery", val_gallery)
        print_stats("query", query)
        print_stats("gallery", gallery)
        print(f"  identity overlap train/test: 0 (checked)")
        print(f"  written to {protocol_root}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
