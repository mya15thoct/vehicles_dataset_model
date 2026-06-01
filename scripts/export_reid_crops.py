#!/usr/bin/env python3
"""Export vehicle crops from CVAT XML annotations for Re-ID experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install pillow") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.json")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--annotation-root", default=None)
    parser.add_argument("--output-root", default="reid_crops")
    parser.add_argument(
        "--completed-only",
        action="store_true",
        help="Only export conditions marked completed.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Skip boxes smaller than this pixel area.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse annotations and print counts without writing crops.",
    )
    return parser.parse_args()


def iter_boxes(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    for image in root.findall("image"):
        frame_id = int(image.attrib["id"])
        frame_name = image.attrib["name"]
        for box_index, box in enumerate(image.findall("box")):
            attr = box.find("attribute[@name='id']")
            if attr is None or not attr.text or not attr.text.strip():
                continue
            yield {
                "frame_id": frame_id,
                "frame_name": frame_name,
                "box_index": box_index,
                "label": normalize_label(box.attrib.get("label", "")),
                "id": int(attr.text.strip()),
                "xtl": float(box.attrib["xtl"]),
                "ytl": float(box.attrib["ytl"]),
                "xbr": float(box.attrib["xbr"]),
                "ybr": float(box.attrib["ybr"]),
            }


def normalize_label(label: str) -> str:
    return label.strip().lower()


def clamp_box(box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(width, round(box["xtl"])))
    top = max(0, min(height, round(box["ytl"])))
    right = max(0, min(width, round(box["xbr"])))
    bottom = max(0, min(height, round(box["ybr"])))
    return left, top, right, bottom


def resolve_path(root: Path, name: str, fallback_root: Path) -> Path:
    path = root / name
    if path.exists():
        return path
    fallback = fallback_root / name
    if fallback.exists():
        return fallback
    return path


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    image_root = Path(args.image_root or config["image_root"])
    annotation_root = Path(args.annotation_root or config["annotation_root"])
    output_root = Path(args.output_root)
    manifest_rows = []

    total = 0
    skipped = 0
    print(f"Image root: {image_root}")
    print(f"Annotation root: {annotation_root}")
    print(f"Output root: {output_root}")

    for condition in config["conditions"]:
        if args.completed_only and condition.get("status") != "completed":
            continue

        for view_name, view_cfg in condition["views"].items():
            xml_path = resolve_path(annotation_root, view_cfg["annotation"], repo_root)
            image_dir = resolve_path(image_root, view_cfg["images"], repo_root)

            if not xml_path.exists():
                print(f"skip missing XML: {xml_path}")
                continue
            if not image_dir.exists():
                print(f"skip missing image folder: {image_dir}")
                continue

            view_count = 0
            for box in iter_boxes(xml_path):
                image_path = image_dir / box["frame_name"]
                if not image_path.exists():
                    skipped += 1
                    continue

                area = max(0.0, box["xbr"] - box["xtl"]) * max(0.0, box["ybr"] - box["ytl"])
                if area < args.min_area:
                    skipped += 1
                    continue

                relative_crop = Path(condition["name"]) / view_name / f"id_{box['id']:06d}"
                crop_name = (
                    f"{condition['name']}_{view_name}_id{box['id']:06d}_"
                    f"frame{box['frame_id']:06d}_{box['label']}.jpg"
                )
                crop_path = output_root / relative_crop / crop_name

                if not args.dry_run:
                    with Image.open(image_path) as image:
                        left, top, right, bottom = clamp_box(box, image.width, image.height)
                        if right <= left or bottom <= top:
                            skipped += 1
                            continue
                        crop_path.parent.mkdir(parents=True, exist_ok=True)
                        image.crop((left, top, right, bottom)).save(crop_path, quality=95)

                manifest_rows.append(
                    {
                        "condition": condition["name"],
                        "view": view_name,
                        "vehicle_id": box["id"],
                        "label": box["label"],
                        "frame_id": box["frame_id"],
                        "frame_name": box["frame_name"],
                        "source_image": str(image_path),
                        "crop_path": str(crop_path),
                        "xtl": box["xtl"],
                        "ytl": box["ytl"],
                        "xbr": box["xbr"],
                        "ybr": box["ybr"],
                    }
                )
                total += 1
                view_count += 1

            print(f"{condition['name']} {view_name}: exported={view_count}")

    if not args.dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = output_root / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else [])
            if manifest_rows:
                writer.writeheader()
                writer.writerows(manifest_rows)
        print(f"Manifest: {manifest_path}")

    print(f"Total crops: {total}")
    print(f"Skipped boxes/images: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
