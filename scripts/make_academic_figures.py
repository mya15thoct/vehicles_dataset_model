#!/usr/bin/env python3
"""Paper-ready figures in the conventions journals actually use.

The exploratory charts elsewhere in this repo are styled for reading on a
screen: rounded bars, card containers, titles and subtitles baked into the
image. That is the wrong register for a manuscript. Every figure here follows
the rules IEEE-style papers follow instead:

  - no title or subtitle inside the image; both belong in the LaTeX caption
  - labels sit in the margins as small plain text, not on decorative headers
  - plain rectangular marks and hairline rules, no rounding, no containers
  - white background, tight panel spacing
  - sized to the column width so nothing is rescaled at typesetting time
  - vector PDF for plots, JPEG for photographs

Only figures a table cannot replace belong here. Class counts, condition
counts and identity coverage are Tables 3-5 of the manuscript already; drawing
them again spends a figure slot on redundancy. A size *distribution* is the
exception, because quartiles in a table lose the shape.

Produces:
  fig_dataset_examples        annotated frames, both views x four conditions
  fig_cross_view_pairs        the same vehicle in both views, per condition
  fig_class_examples          vehicle class x condition grid of crops
  fig_crop_size_by_condition  crop size distribution by condition and view
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from PIL import Image

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
VIEW_ORDER = ["before", "after"]
VIEW_LABELS = {"before": "Before (front)", "after": "After (rear)"}
CLASS_ORDER = ["bus", "car", "motorbike", "truck"]
CLASS_LABELS = {"bus": "Bus", "car": "Car", "motorbike": "Motorbike", "truck": "Truck"}

# Okabe-Ito derived: separable in grayscale and under common CVD types.
VIEW_STYLE = {
    "before": {"facecolor": "#4477AA", "label": "Before (front)"},
    "after": {"facecolor": "#EE7733", "label": "After (rear)"},
}
CLASS_COLORS = {
    "bus": "#EE7733",
    "car": "#4477AA",
    "motorbike": "#228833",
    "truck": "#CC3311",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.json")
    parser.add_argument("--manifest", default="/mnt/recover/ngan/vehicles/reid_crops_full/manifest.csv")
    parser.add_argument("--image-root", default="/mnt/recover/ngan/vehicles/multi-weather_traffic_data")
    parser.add_argument("--annotation-root", default="annotation")
    parser.add_argument("--output-root", default="docs/figures/paper")
    parser.add_argument("--font-size", type=float, default=8.0)
    parser.add_argument(
        "--only",
        nargs="+",
        default=["examples", "pairs", "classes", "size"],
        choices=["examples", "pairs", "classes", "size"],
    )
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
            "grid.linewidth": 0.4,
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, output_root: Path, stem: str, photo: bool = False) -> None:
    """Plots ship as vector; photographs as JPEG, where vector only bloats."""
    output_root.mkdir(parents=True, exist_ok=True)
    suffixes = ("jpg",) if photo else ("pdf", "png")
    for suffix in suffixes:
        path = output_root / f"{stem}.{suffix}"
        fig.savefig(path, **({"pil_kwargs": {"quality": 94}} if suffix == "jpg" else {}))
        print(f"Saved {path}")
    plt.close(fig)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------

def normalize_view(view: str) -> str:
    if view.startswith("before"):
        return "before"
    if view.startswith("after"):
        return "after"
    return view


def resolve(root: Path, name: str) -> Path:
    path = root / name
    return path if path.exists() else Path(name)


def parse_xml(xml_path: Path) -> list[dict]:
    records = []
    for image in ET.parse(xml_path).getroot().findall("image"):
        boxes = []
        for box in image.findall("box"):
            attr = box.find("attribute[@name='id']")
            if attr is None or not attr.text or not attr.text.strip():
                continue
            boxes.append(
                {
                    "label": (box.attrib.get("label") or "").strip().lower(),
                    "id": int(attr.text.strip()),
                    "xtl": float(box.attrib["xtl"]),
                    "ytl": float(box.attrib["ytl"]),
                    "xbr": float(box.attrib["xbr"]),
                    "ybr": float(box.attrib["ybr"]),
                }
            )
        records.append({"frame_name": image.attrib["name"], "boxes": boxes})
    return records


def load_streams(config_path: Path, image_root: Path, annotation_root: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    streams = {}
    for condition in config["conditions"]:
        if condition.get("status") != "completed":
            continue
        for view_name, view_cfg in condition["views"].items():
            xml_path = resolve(annotation_root, view_cfg["annotation"])
            image_dir = resolve(image_root, view_cfg["images"])
            if not xml_path.exists():
                print(f"skip missing XML: {xml_path}")
                continue
            streams[(condition["name"], view_name)] = {
                "image_dir": image_dir,
                "records": parse_xml(xml_path),
            }
    return streams


def box_area(box: dict) -> float:
    return max(0.0, box["xbr"] - box["xtl"]) * max(0.0, box["ybr"] - box["ytl"])


def load_crop(image_path: Path, box: dict, pad: float = 0.10) -> Image.Image | None:
    if not image_path.exists():
        return None
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        pad_x = (box["xbr"] - box["xtl"]) * pad
        pad_y = (box["ybr"] - box["ytl"]) * pad
        left = max(0, int(box["xtl"] - pad_x))
        top = max(0, int(box["ytl"] - pad_y))
        right = min(width, int(box["xbr"] + pad_x))
        bottom = min(height, int(box["ybr"] + pad_y))
        if right <= left or bottom <= top:
            return None
        return image.crop((left, top, right, bottom))


def downscale(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    ratio = max_side / max(image.size)
    return image.resize((max(1, int(image.width * ratio)), max(1, int(image.height * ratio))), Image.LANCZOS)


def pad_square(image: Image.Image, side: int, fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Letterbox a crop into a square cell.

    Vehicle crops here are tall: median height/width is 1.68 overall and 2.32
    for motorbikes. Laying tall crops out directly makes a 4x4 gallery roughly
    three times taller than it is wide, which no page can hold. Square cells
    keep the grid predictable and make crops comparable across cells, at the
    cost of some background padding.
    """
    fitted = image.copy()
    fitted.thumbnail((side, side), Image.LANCZOS)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(fitted, ((side - fitted.width) // 2, (side - fitted.height) // 2))
    return canvas


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def blank(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def grid_figsize(images: list[Image.Image], rows: int, cols: int, header: float = 0.45,
                 footer: float = 0.0, width: float = FULL_WIDTH):
    """Height that matches the panel aspect, so no whitespace is left over.

    The source frames are portrait (1080x1920), so a grid sized by guesswork
    either letterboxes every panel or runs off the page. Deriving the height
    from the median panel aspect keeps the figure exactly as tall as its
    content needs.
    """
    if not images:
        return (width, width * 0.6)
    aspects = sorted(im.height / im.width for im in images)
    aspect = aspects[len(aspects) // 2]
    panel_w = width / cols
    return (width, panel_w * aspect * rows + header + footer)


def make_dataset_examples(streams: dict, output_root: Path) -> None:
    """Annotated frames: rows are the two views, columns the four conditions.

    Merges what used to be two separate figures (raw scenes and annotation
    examples): drawing the boxes on the scene shows both at once and frees a
    figure slot.
    """
    # Load first so the figure can be sized to the real panel aspect.
    panels: dict[tuple[str, str], tuple] = {}
    for view in VIEW_ORDER:
        for condition in CONDITION_ORDER:
            stream = streams.get((condition, view))
            if not stream:
                continue
            # A frame with a handful of vehicles reads better than the busiest
            # one, where overlapping boxes hide the scene.
            candidates = [r for r in stream["records"] if 3 <= len(r["boxes"]) <= 6]
            candidates = candidates or [r for r in stream["records"] if r["boxes"]]
            for record in candidates:
                path = stream["image_dir"] / record["frame_name"]
                if not path.exists():
                    continue
                with Image.open(path) as source:
                    image = source.convert("RGB")
                scale = 700 / max(image.size)
                panels[(condition, view)] = (downscale(image, 700), record, scale)
                break

    fig, axes = plt.subplots(
        2, 4, figsize=grid_figsize([p[0] for p in panels.values()], 2, 4, header=0.45, footer=0.35)
    )

    for row, view in enumerate(VIEW_ORDER):
        for col, condition in enumerate(CONDITION_ORDER):
            ax = axes[row][col]
            blank(ax)
            entry = panels.get((condition, view))
            if entry is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            else:
                image, record, scale = entry
                ax.imshow(image)
                for box in record["boxes"]:
                    ax.add_patch(
                        mpatches.Rectangle(
                            (box["xtl"] * scale, box["ytl"] * scale),
                            (box["xbr"] - box["xtl"]) * scale,
                            (box["ybr"] - box["ytl"]) * scale,
                            linewidth=0.7,
                            edgecolor=CLASS_COLORS.get(box["label"], "#666666"),
                            facecolor="none",
                        )
                    )
            if row == 0:
                ax.set_title(CONDITION_LABELS[condition], pad=3)
            if col == 0:
                ax.set_ylabel(VIEW_LABELS[view], labelpad=4)

    handles = [mpatches.Patch(facecolor=CLASS_COLORS[c], label=CLASS_LABELS[c]) for c in CLASS_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(wspace=0.03, hspace=0.03, bottom=0.06)
    save(fig, output_root, "fig_dataset_examples", photo=True)


def make_cross_view_pairs(streams: dict, output_root: Path) -> None:
    """The same vehicle seen from both cameras, one column per condition.

    This is the figure that states the problem: the two rows share an identity
    but almost no visible surface.
    """
    chosen_by_condition: dict[str, tuple] = {}
    for condition in CONDITION_ORDER:
        before = streams.get((condition, "before"))
        after = streams.get((condition, "after"))
        chosen = None
        if before and after:
            # index the largest box per identity in each view, then take the
            # identity whose weaker view is still large -- both panels legible.
            def best_by_id(stream):
                best = {}
                for record in stream["records"]:
                    for box in record["boxes"]:
                        area = box_area(box)
                        key = box["id"]
                        if key not in best or area > best[key][0]:
                            best[key] = (area, record, box)
                return best

            b_best = best_by_id(before)
            a_best = best_by_id(after)
            shared = set(b_best) & set(a_best)
            ranked = sorted(shared, key=lambda i: min(b_best[i][0], a_best[i][0]), reverse=True)
            for vehicle_id in ranked[:20]:
                b_crop = load_crop(before["image_dir"] / b_best[vehicle_id][1]["frame_name"], b_best[vehicle_id][2])
                a_crop = load_crop(after["image_dir"] / a_best[vehicle_id][1]["frame_name"], a_best[vehicle_id][2])
                if b_crop is not None and a_crop is not None:
                    chosen = (
                        vehicle_id,
                        pad_square(b_crop, 420),
                        pad_square(a_crop, 420),
                        b_best[vehicle_id][2]["label"],
                    )
                    break
        if chosen is not None:
            chosen_by_condition[condition] = chosen

    # Square cells: 2 rows x 4 columns lands at a 2:1 figure regardless of how
    # tall the underlying crops are.
    fig, axes = plt.subplots(2, 4, figsize=(FULL_WIDTH, FULL_WIDTH / 4 * 2 + 0.6))

    for col, condition in enumerate(CONDITION_ORDER):
        chosen = chosen_by_condition.get(condition)
        for row, view in enumerate(VIEW_ORDER):
            ax = axes[row][col]
            blank(ax)
            if chosen is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
                continue
            vehicle_id, b_crop, a_crop, label = chosen
            ax.imshow(b_crop if view == "before" else a_crop)
            if row == 0:
                ax.set_title(f"{CONDITION_LABELS[condition]}\nID {vehicle_id} ({label})", pad=3)
            if col == 0:
                ax.set_ylabel(VIEW_LABELS[view], labelpad=4)

    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    save(fig, output_root, "fig_cross_view_pairs", photo=True)


def make_class_examples(streams: dict, output_root: Path) -> None:
    """Vehicle class (rows) against condition (columns).

    Cells take the largest annotated crop for the pair, so the example shows a
    clean annotation rather than a marginal one.
    """
    best: dict[tuple[str, str], tuple] = {}
    for (condition, view), stream in streams.items():
        for record in stream["records"]:
            for box in record["boxes"]:
                if box["label"] not in CLASS_ORDER:
                    continue
                key = (box["label"], condition)
                area = box_area(box)
                if area <= 0:
                    continue
                if key not in best or area > best[key][0]:
                    best[key] = (area, stream["image_dir"], record, box)

    crops: dict[tuple[str, str], Image.Image] = {}
    for key, (_, image_dir, record, box) in best.items():
        crop = load_crop(image_dir / record["frame_name"], box)
        if crop is not None:
            crops[key] = pad_square(crop, 360)

    # Square cells keep this a 4x4 square, instead of the ~3:1 tower that raw
    # motorbike crops would produce.
    fig, axes = plt.subplots(4, 4, figsize=(FULL_WIDTH, FULL_WIDTH + 0.45))
    for row, label in enumerate(CLASS_ORDER):
        for col, condition in enumerate(CONDITION_ORDER):
            ax = axes[row][col]
            blank(ax)
            crop = crops.get((label, condition))
            if crop is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.imshow(crop)
            if row == 0:
                ax.set_title(CONDITION_LABELS[condition], pad=3)
            if col == 0:
                ax.set_ylabel(CLASS_LABELS[label], labelpad=4)

    fig.subplots_adjust(wspace=0.03, hspace=0.05)
    save(fig, output_root, "fig_class_examples", photo=True)


def make_crop_size_figure(manifest_path: Path, output_root: Path) -> None:
    sizes: dict[tuple[str, str], list[float]] = defaultdict(list)
    with manifest_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            width = max(0.0, float(row["xbr"]) - float(row["xtl"]))
            height = max(0.0, float(row["ybr"]) - float(row["ytl"]))
            sizes[(row["condition"], normalize_view(row["view"]))].append(math.sqrt(width * height))
    print(f"Loaded {sum(len(v) for v in sizes.values()):,} crop sizes")

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 2.2))
    positions, data, colors, ticks = [], [], [], []
    for index, condition in enumerate(CONDITION_ORDER):
        for offset, view in enumerate(VIEW_ORDER):
            values = sizes.get((condition, view), [])
            if not values:
                continue
            positions.append(index + (offset - 0.5) * 0.32)
            data.append(values)
            colors.append(VIEW_STYLE[view]["facecolor"])
        ticks.append(index)

    parts = ax.boxplot(
        data,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,  # 100k outlier points would bury the boxes
        medianprops={"color": "black", "linewidth": 0.9},
        whiskerprops={"linewidth": 0.6},
        capprops={"linewidth": 0.6},
        boxprops={"linewidth": 0.6},
    )
    for patch, color in zip(parts["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    ax.set_xticks(ticks)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_ylabel("Crop size (px, equal-area edge)")
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(facecolor=VIEW_STYLE[v]["facecolor"], alpha=0.85, edgecolor="black", linewidth=0.6,
                       label=VIEW_STYLE[v]["label"])
        for v in VIEW_ORDER
    ]
    ax.legend(handles=handles, loc="upper left", ncol=2)
    save(fig, output_root, "fig_crop_size_by_condition")


def main() -> int:
    args = parse_args()
    apply_style(args.font_size)
    output_root = Path(args.output_root)

    needs_images = {"examples", "pairs", "classes"} & set(args.only)
    streams = {}
    if needs_images:
        streams = load_streams(Path(args.config), Path(args.image_root), Path(args.annotation_root))
        print(f"Loaded {len(streams)} annotated streams")

    if "examples" in args.only:
        make_dataset_examples(streams, output_root)
    if "pairs" in args.only:
        make_cross_view_pairs(streams, output_root)
    if "classes" in args.only:
        make_class_examples(streams, output_root)
    if "size" in args.only:
        make_crop_size_figure(Path(args.manifest), output_root)

    print()
    print("No figure carries a title -- write it in the LaTeX caption.")
    print("Use the PDF for the plot and the JPEG for the photograph grids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
