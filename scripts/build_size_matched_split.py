#!/usr/bin/env python3
"""Build a size-matched evaluation split so per-condition results are comparable.

Vehicles too small or too distant to be identified reliably by eye were not
annotated. That is standard practice, but because visibility depends on
illumination and rain, the effective minimum annotated size is not constant:
it runs from about 40 px in morning/no-rain to about 130 px in evening/rain.
The four per-condition test sets are therefore filtered at different
thresholds, and comparing their accuracy directly measures the thresholds as
much as the conditions.

This applies one common threshold to every condition, so what is left is
comparable across conditions. Retraining is not required -- only re-running
evaluation on the filtered CSVs.

The split CSVs carry no box geometry, so sizes are recovered by joining on
crop_path against the crop manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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

SPLIT_NAMES = ["query", "gallery", "val_query", "val_gallery"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/mnt/recover/ngan/vehicles/reid_crops_full/manifest.csv")
    parser.add_argument("--split-root", default="/mnt/recover/ngan/vehicles/reid_benchmark_identity_full")
    parser.add_argument("--output-root", default="/mnt/recover/ngan/vehicles/reid_benchmark_size_matched")
    parser.add_argument(
        "--min-size",
        default="auto",
        help="Equal-area edge length in pixels to keep, or 'auto' to use the "
        "strictest per-stream minimum observed in the split (the only threshold "
        "every condition can actually meet).",
    )
    parser.add_argument("--splits", nargs="+", default=SPLIT_NAMES, choices=SPLIT_NAMES)
    parser.add_argument("--report", default="docs/size_matched_split.md")
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


def load_sizes(manifest_path: Path) -> dict[str, float]:
    sizes: dict[str, float] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            width = max(0.0, float(row["xbr"]) - float(row["xtl"]))
            height = max(0.0, float(row["ybr"]) - float(row["ytl"]))
            sizes[row["crop_path"]] = math.sqrt(width * height)
    return sizes


def identity(row: dict) -> str:
    return f"{row['condition']}::{int(row['vehicle_id']):06d}"


def normalize_view(view: str) -> str:
    if view.startswith("before"):
        return "before"
    if view.startswith("after"):
        return "after"
    return view


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    split_root = Path(args.split_root)
    output_root = Path(args.output_root)

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    sizes = load_sizes(manifest_path)
    print(f"Loaded {len(sizes):,} crop sizes from {manifest_path}")

    loaded: dict[str, list[dict]] = {}
    missing_size = 0
    for name in args.splits:
        path = split_root / f"{name}.csv"
        if not path.exists():
            print(f"skip missing split: {path}")
            continue
        rows = read_csv(path)
        for row in rows:
            row["_size"] = sizes.get(row["crop_path"])
            if row["_size"] is None:
                missing_size += 1
        loaded[name] = rows

    if missing_size:
        print(
            f"WARNING: {missing_size} split rows had no matching crop_path in the manifest "
            "and will be dropped. Check that the manifest and the split were built from the "
            "same crop export."
        )

    # The only threshold every condition can meet is the largest per-stream
    # minimum: anything lower leaves at least one condition unfiltered while
    # others are cut, which is the confound this is meant to remove.
    per_stream_min: dict[tuple[str, str], float] = {}
    for rows in loaded.values():
        for row in rows:
            if row["_size"] is None:
                continue
            key = (row["condition"], normalize_view(row["view"]))
            current = per_stream_min.get(key)
            if current is None or row["_size"] < current:
                per_stream_min[key] = row["_size"]

    if args.min_size == "auto":
        threshold = max(per_stream_min.values()) if per_stream_min else 0.0
        print(f"auto threshold = {threshold:.1f} px (strictest per-stream minimum)")
    else:
        threshold = float(args.min_size)
        print(f"threshold = {threshold:.1f} px (given)")

    print()
    print("per-stream minimum size in the original split:")
    for key in sorted(per_stream_min):
        print(f"  {key[0]:16s} {key[1]:7s} {per_stream_min[key]:7.1f} px")

    report_rows = []
    filtered: dict[str, list[dict]] = {}
    for name, rows in loaded.items():
        kept = [r for r in rows if r["_size"] is not None and r["_size"] >= threshold]
        filtered[name] = kept
        write_csv(output_root / f"{name}.csv", kept)

        by_condition_before = defaultdict(int)
        by_condition_after = defaultdict(int)
        for row in rows:
            by_condition_before[row["condition"]] += 1
        for row in kept:
            by_condition_after[row["condition"]] += 1

        ids_before = {identity(r) for r in rows}
        ids_after = {identity(r) for r in kept}
        print()
        print(
            f"{name}: {len(rows):,} -> {len(kept):,} rows "
            f"({len(kept) / len(rows) * 100 if rows else 0:.1f}% kept), "
            f"identities {len(ids_before):,} -> {len(ids_after):,}"
        )
        for condition in sorted(by_condition_before):
            before = by_condition_before[condition]
            after = by_condition_after.get(condition, 0)
            print(f"    {condition:16s} {before:6,d} -> {after:6,d}  ({after / before * 100 if before else 0:5.1f}%)")
            report_rows.append(
                {"split": name, "condition": condition, "before": before, "after": after}
            )

    # A query whose identity lost every gallery crop can never be matched; the
    # metric silently skips it, so it has to be counted here instead.
    for query_name, gallery_name in (("query", "gallery"), ("val_query", "val_gallery")):
        if query_name not in filtered or gallery_name not in filtered:
            continue
        gallery_ids = {identity(r) for r in filtered[gallery_name]}
        orphan = {identity(r) for r in filtered[query_name]} - gallery_ids
        print()
        if orphan:
            print(
                f"NOTE: {len(orphan)} {query_name} identities have no surviving {gallery_name} "
                "crop after filtering. They are unmatchable and are excluded from the metric "
                "automatically; report the surviving query count alongside the numbers."
            )
        else:
            print(f"every {query_name} identity still has at least one {gallery_name} crop")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Size-Matched Evaluation Split",
        "",
        f"Threshold: **{threshold:.1f} px** equal-area edge length.",
        "",
        "Vehicles too distant to be identified by eye were not annotated, so the",
        "effective minimum annotated size varies with illumination and rain. Comparing",
        "the four per-condition test sets at their native thresholds would measure the",
        "thresholds as much as the conditions. Applying one common threshold makes the",
        "conditions comparable; no retraining is involved, only re-evaluation.",
        "",
        "## Rows kept per split and condition",
        "",
        "| Split | Condition | Before | After | Kept |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in report_rows:
        pct = row["after"] / row["before"] * 100 if row["before"] else 0.0
        lines.append(
            f"| {row['split']} | {row['condition']} | {row['before']:,} | {row['after']:,} | {pct:.1f}% |"
        )
    lines += [
        "",
        "## Per-stream minimum size in the original split",
        "",
        "| Condition | View | Min size (px) |",
        "| --- | --- | ---: |",
    ]
    for key in sorted(per_stream_min):
        lines.append(f"| {key[0]} | {key[1]} | {per_stream_min[key]:.1f} |")
    lines += [
        "",
        "## How to use",
        "",
        "Re-run `methods/wicv/evaluate.py` with `--query` and `--gallery` pointing at",
        "the filtered CSVs, and report the size-matched per-condition table next to the",
        "native one. Say in the text that the native table is confounded by the",
        "annotation threshold and that the size-matched table is the controlled",
        "comparison.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {report_path}")
    print(f"Saved filtered splits to {output_root}")

    stats_path = output_root / "size_matched_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "threshold": threshold,
                "per_stream_min": {f"{c}|{v}": s for (c, v), s in per_stream_min.items()},
                "rows": report_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
