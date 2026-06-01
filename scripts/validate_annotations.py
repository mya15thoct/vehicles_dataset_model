#!/usr/bin/env python3
"""Validate CVAT XML annotations for the vehicle Re-ID dataset."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/dataset.json",
        help="Path to dataset config JSON.",
    )
    parser.add_argument(
        "--annotation-root",
        default=None,
        help="Override annotation root. Use '.' for local XML working copies.",
    )
    parser.add_argument(
        "--include-in-progress",
        action="store_true",
        help="Also validate conditions marked in_progress if files exist.",
    )
    return parser.parse_args()


def iter_boxes(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    for image in root.findall("image"):
        frame_id = int(image.attrib["id"])
        frame_name = image.attrib["name"]
        for box in image.findall("box"):
            attr = box.find("attribute[@name='id']")
            vehicle_id = attr.text.strip() if attr is not None and attr.text else None
            yield {
                "frame_id": frame_id,
                "frame_name": frame_name,
                "label": box.attrib.get("label", ""),
                "id": int(vehicle_id) if vehicle_id else None,
                "xtl": float(box.attrib["xtl"]),
                "ytl": float(box.attrib["ytl"]),
                "xbr": float(box.attrib["xbr"]),
                "ybr": float(box.attrib["ybr"]),
            }


def load_view(xml_path: Path) -> dict:
    boxes = list(iter_boxes(xml_path))
    labels = Counter(box["label"] for box in boxes)
    missing_id = sum(1 for box in boxes if box["id"] is None)
    invalid_boxes = [
        box
        for box in boxes
        if box["xbr"] <= box["xtl"] or box["ybr"] <= box["ytl"]
    ]

    id_labels: dict[int, Counter] = defaultdict(Counter)
    for box in boxes:
        if box["id"] is not None:
            id_labels[box["id"]][box["label"]] += 1

    mixed_labels = {
        vehicle_id: labels
        for vehicle_id, labels in id_labels.items()
        if len(labels) > 1
    }

    return {
        "boxes": boxes,
        "labels": labels,
        "missing_id": missing_id,
        "invalid_boxes": invalid_boxes,
        "id_labels": id_labels,
        "mixed_labels": mixed_labels,
    }


def resolve_xml_path(annotation_root: Path, annotation_name: str, fallback_root: Path) -> Path:
    xml_path = annotation_root / annotation_name
    if xml_path.exists():
        return xml_path
    fallback = fallback_root / annotation_name
    if fallback.exists():
        return fallback
    return xml_path


def view_group_name(view_name: str) -> str:
    if view_name.startswith("before"):
        return "before"
    if view_name.startswith("after"):
        return "after"
    return view_name


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    repo_root = Path.cwd()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    annotation_root = Path(args.annotation_root or config["annotation_root"])

    had_error = False
    print(f"Annotation root: {annotation_root}")

    for condition in config["conditions"]:
        if condition.get("status") != "completed" and not args.include_in_progress:
            continue

        print(f"\n== {condition['name']} ==")
        views = {}
        for view_name, view_cfg in condition["views"].items():
            xml_path = resolve_xml_path(annotation_root, view_cfg["annotation"], repo_root)
            if not xml_path.exists():
                print(f"  {view_name}: missing XML: {xml_path}")
                had_error = True
                continue

            data = load_view(xml_path)
            views[view_name] = data
            unique_ids = sorted(data["id_labels"])
            print(
                f"  {view_name}: frames_xml={view_cfg['annotation']}, "
                f"boxes={len(data['boxes'])}, ids={len(unique_ids)}, "
                f"missing_id={data['missing_id']}, invalid_boxes={len(data['invalid_boxes'])}, "
                f"labels={dict(sorted(data['labels'].items()))}"
            )

            if data["mixed_labels"]:
                had_error = True
                sample = list(sorted(data["mixed_labels"].items()))[:10]
                formatted = "; ".join(
                    f"id={vehicle_id} labels={dict(labels)}"
                    for vehicle_id, labels in sample
                )
                print(f"    mixed labels inside view: {formatted}")

        grouped = defaultdict(list)
        for view_name, data in views.items():
            grouped[view_group_name(view_name)].append(data)

        if "before" in grouped and "after" in grouped:
            before_labels = merge_id_labels(grouped["before"])
            after_labels = merge_id_labels(grouped["after"])
            before_ids = set(before_labels)
            after_ids = set(after_labels)
            shared_ids = before_ids & after_ids
            mismatches = []
            for vehicle_id in sorted(shared_ids):
                before_set = set(before_labels[vehicle_id])
                after_set = set(after_labels[vehicle_id])
                if before_set != after_set:
                    mismatches.append((vehicle_id, before_set, after_set))

            print(
                f"  cross-view: before_ids={len(before_ids)}, after_ids={len(after_ids)}, "
                f"shared={len(shared_ids)}, before_only={len(before_ids - after_ids)}, "
                f"after_only={len(after_ids - before_ids)}, label_mismatch={len(mismatches)}"
            )

            if mismatches:
                had_error = True
                sample = mismatches[:10]
                formatted = "; ".join(
                    f"id={vehicle_id} before={sorted(before_set)} after={sorted(after_set)}"
                    for vehicle_id, before_set, after_set in sample
                )
                print(f"    mismatch sample: {formatted}")

    return 1 if had_error else 0


def merge_id_labels(view_datas: list[dict]) -> dict[int, Counter]:
    merged: dict[int, Counter] = defaultdict(Counter)
    for data in view_datas:
        for vehicle_id, labels in data["id_labels"].items():
            merged[vehicle_id].update(labels)
    return merged


if __name__ == "__main__":
    sys.exit(main())
