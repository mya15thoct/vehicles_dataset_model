#!/usr/bin/env python3
"""Paper-ready figures in the conventions journals actually use.

The exploratory charts elsewhere in this repo are styled for reading on a
screen: rounded bars, card containers, titles and subtitles baked into the
image. That is the wrong register for a manuscript. Here the rules are the
ones IEEE-style papers follow:

  - no title inside the image; it belongs in the LaTeX caption
  - axes carry labels and units, with visible ticks
  - plain rectangular marks, no rounding, no container chrome
  - sized to the column width so no rescaling happens at typesetting time
  - vector PDF output, which is what \\includegraphics should point at

Only figures that a table cannot replace belong here. Class counts, condition
counts and identity coverage are already Tables 3-5 of the manuscript; drawing
them again wastes a figure slot and invites a redundancy comment. A size
*distribution* is the exception: quartiles in a table lose the shape.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE two-column: 3.5 in per column, 7.16 in across both.
COLUMN_WIDTH = 3.5
FULL_WIDTH = 7.16

CONDITION_ORDER = ["morning_norain", "evening_norain", "morning_rain", "evening_rain"]
CONDITION_LABELS = {
    "morning_norain": "Morning\nNo rain",
    "evening_norain": "Evening\nNo rain",
    "morning_rain": "Morning\nRain",
    "evening_rain": "Evening\nRain",
}
# Okabe-Ito pair: distinguishable in grayscale and under common CVD types.
VIEW_STYLE = {
    "before": {"facecolor": "#4477AA", "label": "Before (front)"},
    "after": {"facecolor": "#EE7733", "label": "After (rear)"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="/mnt/recover/ngan/vehicles/reid_crops_full/manifest.csv")
    parser.add_argument("--output-root", default="docs/figures/paper")
    parser.add_argument("--font-size", type=float, default=8.0)
    return parser.parse_args()


def apply_style(font_size: float) -> None:
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "grid.linewidth": 0.4,
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,  # embed TrueType so the PDF is editable/searchable
            "ps.fonttype": 42,
        }
    )


def normalize_view(view: str) -> str:
    if view.startswith("before"):
        return "before"
    if view.startswith("after"):
        return "after"
    return view


def load_sizes(manifest_path: Path) -> dict[tuple[str, str], list[float]]:
    sizes: dict[tuple[str, str], list[float]] = defaultdict(list)
    with manifest_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            width = max(0.0, float(row["xbr"]) - float(row["xtl"]))
            height = max(0.0, float(row["ybr"]) - float(row["ytl"]))
            sizes[(row["condition"], normalize_view(row["view"]))].append(math.sqrt(width * height))
    return sizes


def save(fig, output_root: Path, stem: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output_root / f"{stem}.{suffix}"
        fig.savefig(path)
        print(f"Saved {path}")
    plt.close(fig)


def make_object_size_figure(sizes: dict[tuple[str, str], list[float]], output_root: Path) -> None:
    """Crop size by condition and view.

    Box plots rather than a histogram: the comparison the text needs is across
    eight groups, and eight overlaid histograms are unreadable at column width.
    Medians, spread and the before/after offset all stay legible here.
    """
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.4))

    positions = []
    data = []
    colors = []
    tick_positions = []
    for index, condition in enumerate(CONDITION_ORDER):
        base = index * 1.0
        for offset, view in enumerate(("before", "after")):
            values = sizes.get((condition, view), [])
            if not values:
                continue
            positions.append(base + (offset - 0.5) * 0.32)
            data.append(values)
            colors.append(VIEW_STYLE[view]["facecolor"])
        tick_positions.append(base)

    parts = ax.boxplot(
        data,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,  # 100k points of outlier ink would bury the boxes
        medianprops={"color": "black", "linewidth": 0.9},
        whiskerprops={"linewidth": 0.6},
        capprops={"linewidth": 0.6},
        boxprops={"linewidth": 0.6},
    )
    for patch, color in zip(parts["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_ylabel("Crop size (px, equal-area edge)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=VIEW_STYLE[v]["facecolor"], alpha=0.85, edgecolor="black", linewidth=0.6)
        for v in ("before", "after")
    ]
    ax.legend(handles, [VIEW_STYLE[v]["label"] for v in ("before", "after")], loc="upper left", ncol=2)

    save(fig, output_root, "fig_crop_size_by_condition")


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    apply_style(args.font_size)
    sizes = load_sizes(manifest_path)
    total = sum(len(v) for v in sizes.values())
    print(f"Loaded {total:,} crop sizes across {len(sizes)} condition/view streams")

    make_object_size_figure(sizes, Path(args.output_root))

    print()
    print("Use the PDF in \\includegraphics; the PNG is only for quick viewing.")
    print("The figure carries no title -- write it in the LaTeX caption instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
