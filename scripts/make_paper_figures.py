#!/usr/bin/env python3
"""Generate paper figures from dataset images and CVAT annotations."""

from __future__ import annotations

import argparse
import json
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc


PAPER_BG = (248, 249, 250)
CARD_BG = (255, 255, 255)
INK = (25, 31, 40)
MUTED = (91, 99, 112)
GRID = (222, 226, 232)
ACCENT = (36, 95, 145)

CLASS_COLORS = {
    "bus": (215, 103, 33),
    "car": (42, 123, 189),
    "motorbike": (66, 154, 93),
    "truck": (196, 67, 74),
}

CONDITION_COLORS = {
    "morning_norain": (64, 133, 184),
    "evening_norain": (117, 108, 182),
    "morning_rain": (72, 157, 120),
    "evening_rain": (202, 122, 58),
}

VIEW_COLORS = {
    "before": (62, 126, 184),
    "after": (220, 132, 59),
}

CONDITION_LABELS = {
    "morning_norain": "Morning / No Rain",
    "evening_norain": "Evening / No Rain",
    "morning_rain": "Morning / Rain",
    "evening_rain": "Evening / Rain",
}

VIEW_LABELS = {
    "before": "Before View",
    "after": "After View",
}

CLASS_LABELS = {
    "bus": "Bus",
    "car": "Car",
    "motorbike": "Motorbike",
    "truck": "Truck",
}

VIEW_ORDER = ("before", "after")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.json")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--annotation-root", default=None)
    parser.add_argument("--output-root", default="docs/figures")
    parser.add_argument("--max-boxes", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--thumb-width", type=int, default=420)
    parser.add_argument("--crop-size", type=int, default=280)
    return parser.parse_args()


def normalize_label(label: str) -> str:
    return label.strip().lower()


def resolve_path(root: Path, name: str, fallback_root: Path) -> Path:
    path = root / name
    if path.exists():
        return path
    fallback = fallback_root / name
    if fallback.exists():
        return fallback
    return path


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def pretty_condition(name: str) -> str:
    return CONDITION_LABELS.get(name, name.replace("_", " ").title())


def pretty_view(name: str) -> str:
    return VIEW_LABELS.get(name, name.title())


def pretty_class(name: str) -> str:
    return CLASS_LABELS.get(name, name.replace("_", " ").title())


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = INK,
) -> None:
    tw, th = text_size(draw, text, font)
    x1, y1, x2, y2 = box
    draw.text((x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 - th) // 2 - 1), text, fill=fill, font=font)


def save_image(image: Image.Image, path: Path, quality: int = 95) -> None:
    save_kwargs = {"dpi": (300, 300)}
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs.update({"quality": quality, "subsampling": 0})
    image.save(path, **save_kwargs)


def parse_xml(xml_path: Path) -> list[dict]:
    records = []
    root = ET.parse(xml_path).getroot()
    for image in root.findall("image"):
        boxes = []
        for box in image.findall("box"):
            attr = box.find("attribute[@name='id']")
            vehicle_id = None
            if attr is not None and attr.text and attr.text.strip():
                vehicle_id = int(attr.text.strip())
            boxes.append(
                {
                    "label": normalize_label(box.attrib.get("label", "")),
                    "id": vehicle_id,
                    "xtl": float(box.attrib["xtl"]),
                    "ytl": float(box.attrib["ytl"]),
                    "xbr": float(box.attrib["xbr"]),
                    "ybr": float(box.attrib["ybr"]),
                }
            )
        records.append(
            {
                "frame_id": int(image.attrib["id"]),
                "frame_name": image.attrib["name"],
                "width": int(float(image.attrib.get("width", 0))),
                "height": int(float(image.attrib.get("height", 0))),
                "boxes": boxes,
            }
        )
    return records


def load_dataset(config_path: Path, image_root_arg: str | None, annotation_root_arg: str | None) -> list[dict]:
    repo_root = Path.cwd()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    image_root = Path(image_root_arg or config["image_root"])
    annotation_root = Path(annotation_root_arg or config["annotation_root"])

    dataset = []
    for condition in config["conditions"]:
        if condition.get("status") != "completed":
            continue
        condition_item = {"name": condition["name"], "views": {}}
        for view_name, view_cfg in condition["views"].items():
            xml_path = resolve_path(annotation_root, view_cfg["annotation"], repo_root)
            image_dir = resolve_path(image_root, view_cfg["images"], repo_root)
            if not xml_path.exists():
                raise FileNotFoundError(f"Missing annotation XML: {xml_path}")
            records = parse_xml(xml_path)
            condition_item["views"][view_name] = {
                "xml_path": xml_path,
                "image_dir": image_dir,
                "records": records,
            }
        dataset.append(condition_item)
    return dataset


def image_path_for(view_data: dict, record: dict) -> Path:
    return view_data["image_dir"] / record["frame_name"]


def choose_overview_record(view_data: dict) -> dict | None:
    records = view_data["records"]
    if not records:
        return None
    midpoint = (len(records) - 1) / 2
    candidates = []
    for index, record in enumerate(records):
        path = image_path_for(view_data, record)
        if not path.exists():
            continue
        box_count = len(record["boxes"])
        if box_count == 0:
            continue
        balance = 1.0 - abs(index - midpoint) / max(1.0, midpoint)
        score = min(box_count, 12) * 100.0 + balance
        candidates.append((score, index, record))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[2]


def choose_annotation_record(view_data: dict, target_boxes: int = 9) -> dict | None:
    candidates = []
    for index, record in enumerate(view_data["records"]):
        path = image_path_for(view_data, record)
        if not path.exists():
            continue
        box_count = len(record["boxes"])
        if box_count == 0:
            continue
        score = -abs(box_count - target_boxes) + min(box_count, target_boxes) * 0.2
        candidates.append((score, index, record))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[2]


def fit_image(image: Image.Image, width: int, height: int, fill: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized = image.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), fill)
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def frame_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    panel = Image.new("RGB", (width, height), CARD_BG)
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=12, fill=CARD_BG, outline=GRID, width=1)
    fitted = fit_image(image, width - 18, height - 18, fill=CARD_BG)
    panel.paste(fitted, (9, 9))
    return panel


def draw_boxes_on_image(
    image_path: Path,
    boxes: list[dict],
    output_width: int,
    output_height: int,
    max_boxes: int,
) -> Image.Image:
    label_font = load_font(16, bold=True)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        scale = min(output_width / image.width, output_height / image.height)
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (output_width, output_height), CARD_BG)
    offset_x = (output_width - image.width) // 2
    offset_y = (output_height - image.height) // 2
    canvas.paste(image, (offset_x, offset_y))
    draw = ImageDraw.Draw(canvas)
    sorted_boxes = sorted(
        boxes,
        key=lambda item: (item["xbr"] - item["xtl"]) * (item["ybr"] - item["ytl"]),
        reverse=True,
    )[:max_boxes]

    for box in sorted_boxes:
        label = box["label"]
        color = CLASS_COLORS.get(label, (120, 120, 120))
        x1 = offset_x + int(box["xtl"] * scale)
        y1 = offset_y + int(box["ytl"] * scale)
        x2 = offset_x + int(box["xbr"] * scale)
        y2 = offset_y + int(box["ybr"] * scale)
        for line_offset in range(3):
            draw.rectangle((x1 - line_offset, y1 - line_offset, x2 + line_offset, y2 + line_offset), outline=color)
        text = f"{pretty_class(label)} #{box['id']}" if box["id"] is not None else pretty_class(label)
        tw, th = text_size(draw, text, label_font)
        label_y = max(0, y1 - th - 6)
        draw.rounded_rectangle((x1, label_y, x1 + tw + 10, label_y + th + 7), radius=4, fill=color)
        draw.text((x1 + 4, label_y + 2), text, fill=(255, 255, 255), font=label_font)

    panel = Image.new("RGB", (output_width + 18, output_height + 18), CARD_BG)
    panel_draw = ImageDraw.Draw(panel)
    panel_draw.rounded_rectangle((0, 0, panel.width - 1, panel.height - 1), radius=12, fill=CARD_BG, outline=GRID, width=1)
    panel.paste(canvas, (9, 9))
    return panel


def make_grid(images: list[Image.Image], rows: int, cols: int, gap: int = 18, bg=(255, 255, 255)) -> Image.Image:
    if not images:
        raise ValueError("No images to compose")
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    canvas = Image.new("RGB", (cols * cell_w + (cols + 1) * gap, rows * cell_h + (rows + 1) * gap), bg)
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = gap + col * (cell_w + gap) + (cell_w - image.width) // 2
        y = gap + row * (cell_h + gap) + (cell_h - image.height) // 2
        canvas.paste(image, (x, y))
    return canvas


def make_two_view_table(
    rows: list[dict],
    cell_width: int,
    cell_height: int,
    title: str,
    subtitle: str,
) -> Image.Image:
    margin = 34
    label_w = 245
    gap = 16
    title_h = 74
    header_h = 52
    row_h = cell_height + 18
    width = margin * 2 + label_w + gap + 2 * cell_width + gap
    height = margin + title_h + header_h + len(rows) * row_h + margin
    canvas = Image.new("RGB", (width, height), PAPER_BG)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(28, bold=True)
    subtitle_font = load_font(16)
    header_font = load_font(19, bold=True)
    row_font = load_font(18, bold=True)

    draw.text((margin, margin - 2), title, fill=INK, font=title_font)
    draw.text((margin, margin + 36), subtitle, fill=MUTED, font=subtitle_font)

    y = margin + title_h
    x_before = margin + label_w + gap
    x_after = x_before + cell_width + gap
    draw_centered_text(draw, (x_before, y, x_before + cell_width, y + header_h), "Before view", header_font, INK)
    draw_centered_text(draw, (x_after, y, x_after + cell_width, y + header_h), "After view", header_font, INK)

    y += header_h
    for row in rows:
        condition = row["condition"]
        color = CONDITION_COLORS.get(condition, ACCENT)
        draw.rounded_rectangle((margin, y + 9, margin + label_w - 10, y + row_h - 9), radius=12, fill=CARD_BG, outline=GRID, width=1)
        draw.rounded_rectangle((margin + 16, y + 25, margin + 24, y + row_h - 25), radius=4, fill=color)
        draw.text((margin + 36, y + row_h // 2 - 14), pretty_condition(condition), fill=INK, font=row_font)
        canvas.paste(row["before"], (x_before, y + 9))
        canvas.paste(row["after"], (x_after, y + 9))
        y += row_h

    return canvas


def make_dataset_overview(dataset: list[dict], output_root: Path, thumb_width: int) -> list[dict]:
    rows = []
    metadata = []
    thumb_height = int(thumb_width * 1.38)
    for condition in dataset:
        row = {"condition": condition["name"]}
        for view in VIEW_ORDER:
            view_data = condition["views"][view]
            record = choose_overview_record(view_data)
            if record is None:
                raise RuntimeError(f"No available overview image for {condition['name']} {view}")
            path = image_path_for(view_data, record)
            with Image.open(path) as image:
                thumb = frame_panel(image.convert("RGB"), thumb_width, thumb_height)
            row[view] = thumb
            metadata.append(
                {
                    "figure": "dataset_overview",
                    "condition": condition["name"],
                    "view": view,
                    "frame": record["frame_name"],
                    "boxes": len(record["boxes"]),
                }
            )
        rows.append(row)
    figure = make_two_view_table(
        rows,
        cell_width=thumb_width,
        cell_height=thumb_height,
        title="Dataset overview",
        subtitle="Representative frames across weather/time conditions and synchronized camera views",
    )
    path = output_root / "figure_01_dataset_overview.jpg"
    save_image(figure, path)
    print(f"Saved {path}")
    return metadata


def make_annotation_examples(dataset: list[dict], output_root: Path, thumb_width: int, max_boxes: int) -> list[dict]:
    rows = []
    metadata = []
    thumb_height = int(thumb_width * 1.38)
    for condition in dataset:
        row = {"condition": condition["name"]}
        for view in VIEW_ORDER:
            view_data = condition["views"][view]
            record = choose_annotation_record(view_data)
            if record is None:
                raise RuntimeError(f"No annotation example for {condition['name']} {view}")
            path = image_path_for(view_data, record)
            row[view] = draw_boxes_on_image(
                path,
                record["boxes"],
                output_width=thumb_width - 18,
                output_height=thumb_height - 18,
                max_boxes=max_boxes,
            )
            metadata.append(
                {
                    "figure": "annotation_examples",
                    "condition": condition["name"],
                    "view": view,
                    "frame": record["frame_name"],
                    "boxes": len(record["boxes"]),
                    "shown_boxes": min(max_boxes, len(record["boxes"])),
                }
            )
        rows.append(row)
    figure = make_two_view_table(
        rows,
        cell_width=thumb_width,
        cell_height=thumb_height,
        title="Annotation examples",
        subtitle="Bounding boxes show class labels and cross-view identity IDs",
    )
    path = output_root / "figure_02_annotation_examples.jpg"
    save_image(figure, path)
    print(f"Saved {path}")
    return metadata


def compute_stats(dataset: list[dict]) -> dict:
    class_counts = Counter()
    condition_counts = Counter()
    view_counts = Counter()
    frame_counts = Counter()
    id_counts = {}
    cross_view = {}

    for condition in dataset:
        condition_name = condition["name"]
        ids_by_view = {}
        for view in VIEW_ORDER:
            view_data = condition["views"][view]
            ids = set()
            for record in view_data["records"]:
                frame_counts[(condition_name, view)] += 1
                for box in record["boxes"]:
                    class_counts[box["label"]] += 1
                    condition_counts[condition_name] += 1
                    view_counts[view] += 1
                    if box["id"] is not None:
                        ids.add(box["id"])
            ids_by_view[view] = ids
            id_counts[(condition_name, view)] = len(ids)
        before_ids = ids_by_view["before"]
        after_ids = ids_by_view["after"]
        cross_view[condition_name] = {
            "before_ids": len(before_ids),
            "after_ids": len(after_ids),
            "shared_ids": len(before_ids & after_ids),
            "before_only_ids": len(before_ids - after_ids),
            "after_only_ids": len(after_ids - before_ids),
        }

    return {
        "class_counts": class_counts,
        "condition_counts": condition_counts,
        "view_counts": view_counts,
        "frame_counts": frame_counts,
        "id_counts": id_counts,
        "cross_view": cross_view,
    }


def draw_horizontal_bar_chart(
    title: str,
    counts: Counter | dict,
    width: int,
    height: int,
    subtitle: str = "",
    color_map: dict | None = None,
    label_map: dict | None = None,
    order: list[str] | None = None,
) -> Image.Image:
    image = Image.new("RGB", (width, height), CARD_BG)
    draw = ImageDraw.Draw(image)
    title_font = load_font(25, bold=True)
    subtitle_font = load_font(15)
    label_font = load_font(17, bold=True)
    value_font = load_font(15)
    small_font = load_font(13)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=18, fill=CARD_BG, outline=GRID, width=1)
    draw.text((30, 24), title, fill=INK, font=title_font)
    if subtitle:
        draw.text((30, 58), subtitle, fill=MUTED, font=subtitle_font)

    if order is not None:
        items = [(key, counts[key]) for key in order if key in counts]
    else:
        items = list(counts.items())
        items.sort(key=lambda item: item[1], reverse=True)
    total = sum(value for _, value in items)
    max_value = max((value for _, value in items), default=1)
    chart_x = 238
    chart_y = 96
    row_h = max(42, int((height - chart_y - 44) / max(1, len(items))))
    bar_max_w = width - chart_x - 220

    for idx, (label, value) in enumerate(items):
        y = chart_y + idx * row_h
        pct = 100.0 * value / total if total else 0.0
        label_text = label_map.get(label, str(label)) if label_map else str(label)
        color = color_map.get(label, ACCENT) if color_map else ACCENT
        draw.text((30, y + 6), label_text, fill=INK, font=label_font)
        bar_w = int(bar_max_w * value / max_value)
        track_y1 = y + 8
        track_y2 = y + row_h - 11
        draw.rounded_rectangle((chart_x, track_y1, chart_x + bar_max_w, track_y2), radius=9, fill=(238, 241, 245))
        draw.rounded_rectangle((chart_x, track_y1, chart_x + bar_w, track_y2), radius=9, fill=color)
        draw.text(
            (chart_x + bar_max_w + 18, y + 6),
            f"{value:,} ({pct:.1f}%)",
            fill=INK,
            font=value_font,
        )
    draw.text((30, height - 30), f"Total: {total:,}", fill=MUTED, font=small_font)
    return image


def make_statistics_figure(dataset: list[dict], output_root: Path) -> dict:
    stats = compute_stats(dataset)
    panel_w = 980
    panel_h = 340
    panels = [
        draw_horizontal_bar_chart(
            "Vehicle class distribution",
            stats["class_counts"],
            panel_w,
            panel_h,
            subtitle="Number of annotated vehicle boxes per class",
            color_map=CLASS_COLORS,
            label_map=CLASS_LABELS,
            order=["truck", "motorbike", "car", "bus"],
        ),
        draw_horizontal_bar_chart(
            "Weather and time distribution",
            stats["condition_counts"],
            panel_w,
            panel_h,
            subtitle="Annotated boxes across the four collection conditions",
            color_map=CONDITION_COLORS,
            label_map=CONDITION_LABELS,
            order=["morning_norain", "evening_norain", "morning_rain", "evening_rain"],
        ),
        draw_horizontal_bar_chart(
            "Camera-view distribution",
            stats["view_counts"],
            panel_w,
            250,
            subtitle="Annotated boxes in the synchronized before/after views",
            color_map=VIEW_COLORS,
            label_map=VIEW_LABELS,
            order=["before", "after"],
        ),
        draw_horizontal_bar_chart(
            "Shared cross-view identities",
            Counter({key: value["shared_ids"] for key, value in stats["cross_view"].items()}),
            panel_w,
            panel_h,
            subtitle="Identities appearing in both camera views",
            color_map=CONDITION_COLORS,
            label_map=CONDITION_LABELS,
            order=["morning_norain", "evening_norain", "morning_rain", "evening_rain"],
        ),
    ]
    separate_names = [
        "figure_03a_class_distribution.png",
        "figure_03b_condition_distribution.png",
        "figure_03c_view_distribution.png",
        "figure_03d_shared_identities.png",
    ]
    for panel, name in zip(panels, separate_names):
        panel_path = output_root / name
        save_image(panel, panel_path)
        print(f"Saved {panel_path}")

    title_h = 86
    figure = Image.new("RGB", (panel_w * 2 + 42, title_h + panel_h * 2 + 70), PAPER_BG)
    draw = ImageDraw.Draw(figure)
    title_font = load_font(30, bold=True)
    subtitle_font = load_font(16)
    draw.text((28, 24), "Dataset statistics", fill=INK, font=title_font)
    draw.text((28, 62), "Distribution of annotations by class, condition, camera view, and shared identities", fill=MUTED, font=subtitle_font)
    figure.paste(panels[0], (20, title_h))
    figure.paste(panels[1], (panel_w + 22, title_h))
    figure.paste(panels[2], (20, title_h + panel_h + 22))
    figure.paste(panels[3], (panel_w + 22, title_h + panel_h + 22))
    path = output_root / "figure_03_dataset_statistics.png"
    save_image(figure, path)
    print(f"Saved {path}")

    json_stats = {
        "class_counts": dict(stats["class_counts"]),
        "condition_counts": dict(stats["condition_counts"]),
        "view_counts": dict(stats["view_counts"]),
        "cross_view": stats["cross_view"],
        "total_boxes": sum(stats["class_counts"].values()),
    }
    stats_path = output_root / "figure_statistics.json"
    stats_path.write_text(json.dumps(json_stats, indent=2), encoding="utf-8")
    print(f"Saved {stats_path}")
    return json_stats


def crop_box(image_path: Path, box: dict, crop_size: int) -> Image.Image:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        x1 = max(0, int(round(box["xtl"])))
        y1 = max(0, int(round(box["ytl"])))
        x2 = min(width, int(round(box["xbr"])))
        y2 = min(height, int(round(box["ybr"])))
        pad_x = int((x2 - x1) * 0.12)
        pad_y = int((y2 - y1) * 0.12)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        crop = image.crop((x1, y1, x2, y2))
    return fit_image(crop, crop_size, crop_size)


def box_area(box: dict) -> float:
    return max(0.0, box["xbr"] - box["xtl"]) * max(0.0, box["ybr"] - box["ytl"])


def boxes_by_id(view_data: dict) -> dict[int, list[tuple[dict, dict]]]:
    result: dict[int, list[tuple[dict, dict]]] = defaultdict(list)
    for record in view_data["records"]:
        path = image_path_for(view_data, record)
        if not path.exists():
            continue
        for box in record["boxes"]:
            if box["id"] is not None:
                result[box["id"]].append((record, box))
    return result


def best_record_box(items: list[tuple[dict, dict]]) -> tuple[dict, dict]:
    return max(items, key=lambda item: box_area(item[1]))


def choose_shared_identity_pair(condition: dict) -> dict | None:
    before_items = boxes_by_id(condition["views"]["before"])
    after_items = boxes_by_id(condition["views"]["after"])
    shared_ids = sorted(set(before_items) & set(after_items))
    candidates = []
    for vehicle_id in shared_ids:
        before_record, before_box = best_record_box(before_items[vehicle_id])
        after_record, after_box = best_record_box(after_items[vehicle_id])
        score = min(box_area(before_box), box_area(after_box))
        if score <= 0:
            continue
        candidates.append((score, vehicle_id, before_record, before_box, after_record, after_box))
    if not candidates:
        return None
    _, vehicle_id, before_record, before_box, after_record, after_box = max(candidates, key=lambda item: item[0])
    return {
        "id": vehicle_id,
        "label": before_box["label"],
        "before_record": before_record,
        "before_box": before_box,
        "after_record": after_record,
        "after_box": after_box,
    }


def make_cross_view_pairs(dataset: list[dict], output_root: Path, crop_size: int) -> list[dict]:
    title_font = load_font(20, bold=True)
    label_font = load_font(15, bold=True)
    small_font = load_font(14)
    panels = []
    metadata = []
    for condition in dataset:
        pair = choose_shared_identity_pair(condition)
        if pair is None:
            continue
        before_path = image_path_for(condition["views"]["before"], pair["before_record"])
        after_path = image_path_for(condition["views"]["after"], pair["after_record"])
        before_crop = crop_box(before_path, pair["before_box"], crop_size)
        after_crop = crop_box(after_path, pair["after_box"], crop_size)

        panel_w = crop_size * 2 + 54
        panel_h = crop_size + 112
        panel = Image.new("RGB", (panel_w, panel_h), CARD_BG)
        draw = ImageDraw.Draw(panel)
        color = CONDITION_COLORS.get(condition["name"], ACCENT)
        draw.rounded_rectangle((0, 0, panel_w - 1, panel_h - 1), radius=16, fill=CARD_BG, outline=GRID, width=1)
        draw.rounded_rectangle((18, 18, 30, 48), radius=5, fill=color)
        draw.text((42, 16), pretty_condition(condition["name"]), fill=INK, font=title_font)
        draw.text((42, 43), f"ID {pair['id']} | {pretty_class(pair['label'])}", fill=MUTED, font=small_font)
        panel.paste(before_crop, (18, 72))
        panel.paste(after_crop, (crop_size + 36, 72))
        draw.text((18, crop_size + 78), "Before", fill=VIEW_COLORS["before"], font=label_font)
        draw.text((crop_size + 36, crop_size + 78), "After", fill=VIEW_COLORS["after"], font=label_font)
        panels.append(panel)
        metadata.append(
            {
                "figure": "cross_view_pairs",
                "condition": condition["name"],
                "id": pair["id"],
                "label": pair["label"],
                "before_frame": pair["before_record"]["frame_name"],
                "after_frame": pair["after_record"]["frame_name"],
            }
        )

    title_h = 82
    grid = make_grid(panels, rows=2, cols=2, gap=18, bg=PAPER_BG)
    figure = Image.new("RGB", (grid.width + 48, grid.height + title_h + 28), PAPER_BG)
    draw = ImageDraw.Draw(figure)
    figure_title_font = load_font(30, bold=True)
    figure_subtitle_font = load_font(16)
    draw.text((24, 22), "Cross-view positive pairs", fill=INK, font=figure_title_font)
    draw.text((24, 60), "Same identity shown in before/after camera views", fill=MUTED, font=figure_subtitle_font)
    figure.paste(grid, (24, title_h))
    path = output_root / "figure_04_cross_view_positive_pairs.jpg"
    save_image(figure, path)
    print(f"Saved {path}")
    return metadata


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(Path(args.config), args.image_root, args.annotation_root)
    metadata = []
    metadata.extend(make_dataset_overview(dataset, output_root, args.thumb_width))
    metadata.extend(make_annotation_examples(dataset, output_root, args.thumb_width, args.max_boxes))
    stats = make_statistics_figure(dataset, output_root)
    metadata.extend(make_cross_view_pairs(dataset, output_root, args.crop_size))

    metadata_path = output_root / "figure_metadata.json"
    metadata_path.write_text(json.dumps({"selected_examples": metadata, "stats": stats}, indent=2), encoding="utf-8")
    print(f"Saved {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
