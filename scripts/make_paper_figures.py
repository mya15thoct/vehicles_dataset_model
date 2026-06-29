#!/usr/bin/env python3
"""Generate paper figures from dataset images and CVAT annotations."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc


CLASS_COLORS = {
    "bus": (230, 97, 1),
    "car": (50, 136, 189),
    "motorbike": (102, 189, 99),
    "truck": (213, 62, 79),
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


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


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


def add_caption(image: Image.Image, caption: str, font: ImageFont.ImageFont) -> Image.Image:
    draw = ImageDraw.Draw(image)
    caption_h = 42
    canvas = Image.new("RGB", (image.width, image.height + caption_h), (255, 255, 255))
    canvas.paste(image, (0, caption_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, image.width, caption_h), fill=(245, 245, 245))
    tw, th = text_size(draw, caption, font)
    draw.text(((image.width - tw) // 2, (caption_h - th) // 2 - 1), caption, fill=(20, 20, 20), font=font)
    return canvas


def draw_boxes_on_image(
    image_path: Path,
    boxes: list[dict],
    output_width: int,
    max_boxes: int,
    caption: str,
) -> Image.Image:
    font = load_font(18)
    label_font = load_font(15)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        scale = output_width / image.width
        output_height = max(1, int(image.height * scale))
        image = image.resize((output_width, output_height), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(image)
    sorted_boxes = sorted(
        boxes,
        key=lambda item: (item["xbr"] - item["xtl"]) * (item["ybr"] - item["ytl"]),
        reverse=True,
    )[:max_boxes]

    for box in sorted_boxes:
        label = box["label"]
        color = CLASS_COLORS.get(label, (120, 120, 120))
        x1 = int(box["xtl"] * scale)
        y1 = int(box["ytl"] * scale)
        x2 = int(box["xbr"] * scale)
        y2 = int(box["ybr"] * scale)
        for offset in range(3):
            draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
        text = f"{label} #{box['id']}" if box["id"] is not None else label
        tw, th = text_size(draw, text, label_font)
        label_y = max(0, y1 - th - 6)
        draw.rectangle((x1, label_y, x1 + tw + 8, label_y + th + 6), fill=color)
        draw.text((x1 + 4, label_y + 2), text, fill=(255, 255, 255), font=label_font)

    return add_caption(image, caption, font)


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


def make_dataset_overview(dataset: list[dict], output_root: Path, thumb_width: int) -> list[dict]:
    font = load_font(20)
    thumbs = []
    metadata = []
    thumb_height = int(thumb_width * 1.45)
    for condition in dataset:
        for view in VIEW_ORDER:
            view_data = condition["views"][view]
            record = choose_overview_record(view_data)
            if record is None:
                raise RuntimeError(f"No available overview image for {condition['name']} {view}")
            path = image_path_for(view_data, record)
            with Image.open(path) as image:
                thumb = fit_image(image, thumb_width, thumb_height)
            caption = f"{condition['name']} / {view}"
            thumbs.append(add_caption(thumb, caption, font))
            metadata.append(
                {
                    "figure": "dataset_overview",
                    "condition": condition["name"],
                    "view": view,
                    "frame": record["frame_name"],
                    "boxes": len(record["boxes"]),
                }
            )
    figure = make_grid(thumbs, rows=len(dataset), cols=2)
    path = output_root / "figure_01_dataset_overview.jpg"
    figure.save(path, quality=95)
    print(f"Saved {path}")
    return metadata


def make_annotation_examples(dataset: list[dict], output_root: Path, thumb_width: int, max_boxes: int) -> list[dict]:
    panels = []
    metadata = []
    for condition in dataset:
        for view in VIEW_ORDER:
            view_data = condition["views"][view]
            record = choose_annotation_record(view_data)
            if record is None:
                raise RuntimeError(f"No annotation example for {condition['name']} {view}")
            path = image_path_for(view_data, record)
            panels.append(
                draw_boxes_on_image(
                    path,
                    record["boxes"],
                    output_width=thumb_width,
                    max_boxes=max_boxes,
                    caption=f"{condition['name']} / {view}",
                )
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
    figure = make_grid(panels, rows=len(dataset), cols=2)
    path = output_root / "figure_02_annotation_examples.jpg"
    figure.save(path, quality=95)
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
    color: tuple[int, int, int],
) -> Image.Image:
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    title_font = load_font(22)
    label_font = load_font(16)
    value_font = load_font(15)
    draw.text((24, 18), title, fill=(20, 20, 20), font=title_font)

    items = list(counts.items())
    items.sort(key=lambda item: item[1], reverse=True)
    total = sum(value for _, value in items)
    max_value = max((value for _, value in items), default=1)
    chart_x = 165
    chart_y = 64
    row_h = max(34, int((height - chart_y - 24) / max(1, len(items))))
    bar_max_w = width - chart_x - 150

    for idx, (label, value) in enumerate(items):
        y = chart_y + idx * row_h
        pct = 100.0 * value / total if total else 0.0
        draw.text((24, y + 6), str(label), fill=(30, 30, 30), font=label_font)
        bar_w = int(bar_max_w * value / max_value)
        draw.rectangle((chart_x, y + 5, chart_x + bar_w, y + row_h - 8), fill=color)
        draw.rectangle((chart_x, y + 5, chart_x + bar_max_w, y + row_h - 8), outline=(215, 215, 215))
        draw.text(
            (chart_x + bar_w + 10, y + 6),
            f"{value:,} ({pct:.1f}%)",
            fill=(30, 30, 30),
            font=value_font,
        )
    return image


def make_statistics_figure(dataset: list[dict], output_root: Path) -> dict:
    stats = compute_stats(dataset)
    panel_w = 760
    panel_h = 290
    panels = [
        draw_horizontal_bar_chart("Vehicle class distribution", stats["class_counts"], panel_w, panel_h, (80, 143, 190)),
        draw_horizontal_bar_chart("Condition distribution", stats["condition_counts"], panel_w, panel_h, (90, 174, 97)),
        draw_horizontal_bar_chart("View distribution", stats["view_counts"], panel_w, panel_h, (230, 126, 34)),
    ]
    figure = make_grid(panels, rows=3, cols=1, gap=12)
    path = output_root / "figure_03_dataset_statistics.png"
    figure.save(path)
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
    title_font = load_font(20)
    label_font = load_font(17)
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

        panel_w = crop_size * 2 + 34
        panel_h = crop_size + 86
        panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))
        draw = ImageDraw.Draw(panel)
        title = f"{condition['name']}  |  ID {pair['id']}  |  {pair['label']}"
        draw.text((16, 12), title, fill=(20, 20, 20), font=title_font)
        panel.paste(before_crop, (12, 52))
        panel.paste(after_crop, (crop_size + 22, 52))
        draw.text((12, crop_size + 58), "before", fill=(60, 60, 60), font=label_font)
        draw.text((crop_size + 22, crop_size + 58), "after", fill=(60, 60, 60), font=label_font)
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

    figure = make_grid(panels, rows=len(panels), cols=1, gap=16)
    path = output_root / "figure_04_cross_view_positive_pairs.jpg"
    figure.save(path, quality=95)
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
