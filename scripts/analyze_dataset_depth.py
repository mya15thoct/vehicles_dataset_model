#!/usr/bin/env python3
"""Deeper dataset characterization for the paper's dataset section.

The condition/class/identity tables already in docs/dataset_statistics.md say
how much data there is. They do not say how *hard* it is, and they do not back
up the "dense, motorbike-heavy mixed traffic" claim with a number. This script
computes the three distributions that do, all from the crop manifest alone --
no GPU, no retraining:

  1. Object size          -- how small the crops actually are, per condition.
                             Small rear-view crops at night are the mechanism
                             behind the per-condition accuracy gap, so this
                             turns an observation into an explanation.
  2. Scene density        -- annotated instances per frame. This is the number
                             that substantiates "dense traffic"; without it the
                             claim is unsupported.
  3. Instances per identity -- how many crops each vehicle contributes, per
                             view. Re-ID specific: it sets how much evidence
                             retrieval has per query and per gallery entry.

Outputs a JSON blob, a paste-ready Markdown table file, and three figures in
the same flat-white house style as the other paper figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

INK = (25, 31, 40)
MUTED = (91, 99, 112)
GRID = (222, 226, 232)
BG = (255, 255, 255)
ACCENT = (36, 95, 145)

CONDITION_ORDER = ["morning_norain", "evening_norain", "morning_rain", "evening_rain"]
CONDITION_LABELS = {
    "morning_norain": "Morning / No Rain",
    "evening_norain": "Evening / No Rain",
    "morning_rain": "Morning / Rain",
    "evening_rain": "Evening / Rain",
}
CONDITION_COLORS = {
    "morning_norain": (64, 133, 184),
    "evening_norain": (117, 108, 182),
    "morning_rain": (72, 157, 120),
    "evening_rain": (202, 122, 58),
}
VIEW_COLORS = {"before": (62, 126, 184), "after": (220, 132, 59)}

# Size buckets follow the convention used by traffic-dataset papers: the edge
# length of a square of equal area, so "32" means the crop is as big as 32x32.
SIZE_EDGES = [0, 16, 32, 48, 64, 80, 96, 112, 128, 160]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/mnt/recover/ngan/vehicles/reid_crops_full/manifest.csv")
    parser.add_argument("--output-root", default="docs/figures")
    parser.add_argument("--stats-output", default="docs/dataset_depth.md")
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def normalize_view(view: str) -> str:
    if view.startswith("before"):
        return "before"
    if view.startswith("after"):
        return "after"
    return view


def size_bucket_label(index: int) -> str:
    if index == 0:
        return f"<{SIZE_EDGES[1]}"
    if index >= len(SIZE_EDGES) - 1:
        return f">={SIZE_EDGES[-1]}"
    return f"{SIZE_EDGES[index]}-{SIZE_EDGES[index + 1]}"


def size_bucket(edge_length: float) -> int:
    for index in range(len(SIZE_EDGES) - 1, 0, -1):
        if edge_length >= SIZE_EDGES[index]:
            return index
    return 0


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = fraction * (len(sorted_values) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[int(position)]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def analyze(manifest_path: Path) -> dict:
    sizes_all: list[float] = []
    sizes_by_condition: dict[str, list[float]] = defaultdict(list)
    sizes_by_view: dict[str, list[float]] = defaultdict(list)
    size_hist_by_condition: dict[str, Counter] = defaultdict(Counter)

    per_frame: Counter = Counter()
    per_frame_condition: dict[tuple, str] = {}
    per_identity_view: Counter = Counter()
    per_identity: Counter = Counter()
    identity_condition: dict[tuple, str] = {}

    total_rows = 0
    with manifest_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            total_rows += 1
            condition = row["condition"]
            view = normalize_view(row["view"])

            width = max(0.0, float(row["xbr"]) - float(row["xtl"]))
            height = max(0.0, float(row["ybr"]) - float(row["ytl"]))
            edge = math.sqrt(width * height)  # side of an equal-area square
            sizes_all.append(edge)
            sizes_by_condition[condition].append(edge)
            sizes_by_view[view].append(edge)
            size_hist_by_condition[condition][size_bucket(edge)] += 1

            frame_key = (condition, view, row["frame_id"])
            per_frame[frame_key] += 1
            per_frame_condition[frame_key] = condition

            identity_key = (condition, row["vehicle_id"])
            per_identity[identity_key] += 1
            per_identity_view[(condition, row["vehicle_id"], view)] += 1
            identity_condition[identity_key] = condition

    def summarize(values: list[float]) -> dict:
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "mean": sum(ordered) / len(ordered) if ordered else 0.0,
            "median": percentile(ordered, 0.5),
            "p10": percentile(ordered, 0.10),
            "p90": percentile(ordered, 0.90),
            "min": ordered[0] if ordered else 0.0,
            "max": ordered[-1] if ordered else 0.0,
        }

    density_values = list(per_frame.values())
    density_by_condition: dict[str, list[int]] = defaultdict(list)
    for key, count in per_frame.items():
        density_by_condition[per_frame_condition[key]].append(count)

    identity_values = list(per_identity.values())
    identity_by_view: dict[str, list[int]] = defaultdict(list)
    for (_, _, view), count in per_identity_view.items():
        identity_by_view[view].append(count)

    small_16 = sum(1 for value in sizes_all if value < 16)
    small_32 = sum(1 for value in sizes_all if value < 32)

    return {
        "total_boxes": total_rows,
        "object_size": {
            "overall": summarize(sizes_all),
            "by_condition": {c: summarize(v) for c, v in sizes_by_condition.items()},
            "by_view": {v: summarize(vals) for v, vals in sizes_by_view.items()},
            "fraction_below_16px": small_16 / total_rows if total_rows else 0.0,
            "fraction_below_32px": small_32 / total_rows if total_rows else 0.0,
            "histogram_by_condition": {
                c: {size_bucket_label(i): hist.get(i, 0) for i in range(len(SIZE_EDGES))}
                for c, hist in size_hist_by_condition.items()
            },
        },
        "scene_density": {
            "annotated_frames": len(per_frame),
            "overall": summarize([float(v) for v in density_values]),
            "by_condition": {c: summarize([float(x) for x in v]) for c, v in density_by_condition.items()},
            "histogram": dict(Counter(density_values)),
        },
        "instances_per_identity": {
            "num_identities": len(per_identity),
            "overall": summarize([float(v) for v in identity_values]),
            "by_view": {v: summarize([float(x) for x in vals]) for v, vals in identity_by_view.items()},
            "histogram": dict(Counter(identity_values)),
        },
    }


def draw_grouped_bars(
    title: str,
    subtitle: str,
    categories: list[str],
    series: dict[str, list[float]],
    colors: dict[str, tuple],
    value_fmt: str = "{:.0f}",
    width: int = 980,
    height: int = 420,
) -> Image.Image:
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    title_font = load_font(24, bold=True)
    subtitle_font = load_font(14)
    label_font = load_font(13)
    value_font = load_font(11)

    draw.text((24, 20), title, fill=INK, font=title_font)
    draw.text((24, 52), subtitle, fill=MUTED, font=subtitle_font)

    # legend
    x = 24
    y = 78
    for name, color in colors.items():
        draw.rounded_rectangle((x, y, x + 14, y + 14), radius=3, fill=color)
        tw, _ = text_size(draw, name, value_font)
        draw.text((x + 20, y), name, fill=INK, font=value_font)
        x += 20 + tw + 24

    plot_left = 60
    plot_right = width - 30
    plot_top = 112
    plot_bottom = height - 46

    max_value = max((max(vals) for vals in series.values() if vals), default=1.0) or 1.0
    group_w = (plot_right - plot_left) / max(1, len(categories))
    names = list(series.keys())
    bar_w = group_w * 0.7 / max(1, len(names))

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=GRID, width=1)

    for gi, category in enumerate(categories):
        gx = plot_left + gi * group_w
        for si, name in enumerate(names):
            value = series[name][gi] if gi < len(series[name]) else 0.0
            bar_h = (plot_bottom - plot_top) * value / max_value
            bx = gx + group_w * 0.15 + si * bar_w
            draw.rounded_rectangle(
                (bx, plot_bottom - bar_h, bx + bar_w * 0.88, plot_bottom),
                radius=3,
                fill=colors[name],
            )
            if value > 0 and len(names) <= 2:
                label = value_fmt.format(value)
                tw, _ = text_size(draw, label, value_font)
                draw.text((bx + bar_w * 0.44 - tw / 2, plot_bottom - bar_h - 15), label, fill=INK, font=value_font)
        tw, _ = text_size(draw, category, label_font)
        draw.text((gx + group_w / 2 - tw / 2, plot_bottom + 10), category, fill=MUTED, font=label_font)

    return image


def make_size_figure(stats: dict, output_root: Path) -> Path:
    hist = stats["object_size"]["histogram_by_condition"]
    categories = [size_bucket_label(i) for i in range(len(SIZE_EDGES))]
    series = {}
    colors = {}
    for condition in CONDITION_ORDER:
        if condition not in hist:
            continue
        label = CONDITION_LABELS[condition]
        series[label] = [hist[condition].get(c, 0) for c in categories]
        colors[label] = CONDITION_COLORS[condition]

    overall = stats["object_size"]["overall"]
    subtitle = (
        f"Equal-area edge length in pixels. Median {overall['median']:.0f} px; "
        f"{stats['object_size']['fraction_below_32px'] * 100:.1f}% of crops are smaller than 32x32."
    )
    image = draw_grouped_bars(
        "Annotated object size distribution", subtitle, categories, series, colors, width=1080
    )
    path = output_root / "figure_10_object_size.png"
    image.save(path, dpi=(300, 300))
    print(f"Saved {path}")
    return path


def make_density_figure(stats: dict, output_root: Path) -> Path:
    histogram = stats["scene_density"]["histogram"]
    buckets = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "31-40", ">40"]

    def bucket_of(count: int) -> str:
        for index, upper in enumerate([5, 10, 15, 20, 25, 30, 40]):
            if count <= upper:
                return buckets[index]
        return ">40"

    grouped = Counter()
    for count, frames in histogram.items():
        grouped[bucket_of(int(count))] += frames

    overall = stats["scene_density"]["overall"]
    subtitle = (
        f"{stats['scene_density']['annotated_frames']:,} annotated frames; "
        f"mean {overall['mean']:.1f} and median {overall['median']:.0f} vehicles per frame "
        f"(90th percentile {overall['p90']:.0f})."
    )
    image = draw_grouped_bars(
        "Scene density: annotated vehicles per frame",
        subtitle,
        buckets,
        {"frames": [grouped.get(b, 0) for b in buckets]},
        {"frames": ACCENT},
        width=980,
    )
    path = output_root / "figure_11_scene_density.png"
    image.save(path, dpi=(300, 300))
    print(f"Saved {path}")
    return path


def make_identity_figure(stats: dict, output_root: Path) -> Path:
    histogram = stats["instances_per_identity"]["histogram"]
    buckets = ["1-5", "6-10", "11-20", "21-40", "41-70", ">70"]

    def bucket_of(count: int) -> str:
        for index, upper in enumerate([5, 10, 20, 40, 70]):
            if count <= upper:
                return buckets[index]
        return ">70"

    grouped = Counter()
    for count, ids in histogram.items():
        grouped[bucket_of(int(count))] += ids

    overall = stats["instances_per_identity"]["overall"]
    subtitle = (
        f"{stats['instances_per_identity']['num_identities']:,} identities; "
        f"median {overall['median']:.0f} crops per vehicle "
        f"(10th percentile {overall['p10']:.0f}, 90th {overall['p90']:.0f})."
    )
    image = draw_grouped_bars(
        "Instances per vehicle identity",
        subtitle,
        buckets,
        {"identities": [grouped.get(b, 0) for b in buckets]},
        {"identities": ACCENT},
        width=980,
    )
    path = output_root / "figure_12_instances_per_identity.png"
    image.save(path, dpi=(300, 300))
    print(f"Saved {path}")
    return path


def write_markdown(stats: dict, path: Path) -> None:
    size = stats["object_size"]
    density = stats["scene_density"]
    identity = stats["instances_per_identity"]

    lines = [
        "# Dataset Depth Analysis",
        "",
        "Generated by `scripts/analyze_dataset_depth.py` from the crop manifest.",
        "These are the numbers that substantiate the difficulty and density claims",
        "in the dataset section; the existing tables only report volume.",
        "",
        "## Object size (equal-area edge length, pixels)",
        "",
        "| Group | Count | Median | Mean | p10 | p90 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| all | {size['overall']['count']:,} | {size['overall']['median']:.1f} | "
        f"{size['overall']['mean']:.1f} | {size['overall']['p10']:.1f} | {size['overall']['p90']:.1f} |",
    ]
    for view in ("before", "after"):
        if view in size["by_view"]:
            row = size["by_view"][view]
            lines.append(
                f"| view={view} | {row['count']:,} | {row['median']:.1f} | {row['mean']:.1f} | "
                f"{row['p10']:.1f} | {row['p90']:.1f} |"
            )
    for condition in CONDITION_ORDER:
        if condition in size["by_condition"]:
            row = size["by_condition"][condition]
            lines.append(
                f"| {condition} | {row['count']:,} | {row['median']:.1f} | {row['mean']:.1f} | "
                f"{row['p10']:.1f} | {row['p90']:.1f} |"
            )

    lines += [
        "",
        f"Crops smaller than 16x16: {size['fraction_below_16px'] * 100:.2f}%. "
        f"Smaller than 32x32: {size['fraction_below_32px'] * 100:.2f}%.",
        "",
        "## Scene density (annotated vehicles per frame)",
        "",
        "| Group | Frames | Median | Mean | p90 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| all | {density['annotated_frames']:,} | {density['overall']['median']:.1f} | "
        f"{density['overall']['mean']:.2f} | {density['overall']['p90']:.1f} | {density['overall']['max']:.0f} |",
    ]
    for condition in CONDITION_ORDER:
        if condition in density["by_condition"]:
            row = density["by_condition"][condition]
            lines.append(
                f"| {condition} | {row['count']:,} | {row['median']:.1f} | {row['mean']:.2f} | "
                f"{row['p90']:.1f} | {row['max']:.0f} |"
            )

    lines += [
        "",
        "## Instances per vehicle identity",
        "",
        "| Group | Identities | Median | Mean | p10 | p90 | Max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| all | {identity['num_identities']:,} | {identity['overall']['median']:.1f} | "
        f"{identity['overall']['mean']:.2f} | {identity['overall']['p10']:.1f} | "
        f"{identity['overall']['p90']:.1f} | {identity['overall']['max']:.0f} |",
    ]
    for view in ("before", "after"):
        if view in identity["by_view"]:
            row = identity["by_view"][view]
            lines.append(
                f"| view={view} | {row['count']:,} | {row['median']:.1f} | {row['mean']:.2f} | "
                f"{row['p10']:.1f} | {row['p90']:.1f} | {row['max']:.0f} |"
            )

    lines += [
        "",
        "## How to use these in the paper",
        "",
        "- The object-size table is what explains the per-condition accuracy gap:",
        "  quote the median crop size for the hardest condition next to its mAP.",
        "- The density numbers are what back the \"dense mixed traffic\" claim in the",
        "  abstract. Without them that claim is unsupported.",
        "- Instances-per-identity resolves the TODO in the dataset section that asks",
        "  for the true per-identity count instead of the box-count proxy.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {path}")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    stats = analyze(manifest_path)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    make_size_figure(stats, output_root)
    make_density_figure(stats, output_root)
    make_identity_figure(stats, output_root)

    json_path = output_root / "dataset_depth_stats.json"
    json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved {json_path}")

    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(stats, stats_path)

    size = stats["object_size"]["overall"]
    density = stats["scene_density"]["overall"]
    identity = stats["instances_per_identity"]["overall"]
    print()
    print(f"median crop size      : {size['median']:.1f} px")
    print(f"mean vehicles / frame : {density['mean']:.2f}")
    print(f"median crops / identity: {identity['median']:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
