#!/usr/bin/env python3
"""Count boxes directly from the CVAT XML, independent of the crop manifest.

The manifest only keeps boxes that carry an identity attribute, so any box
annotated without an id is invisible to manifest-based statistics. Scene
density computed from the manifest is therefore a lower bound. This script
reads the XML itself and reports both counts side by side, so it is
unambiguous whether the two agree.

If total boxes > boxes with an id, the manifest-derived density understates
the real annotation density and any "how busy is a frame" claim must be
computed from the numbers printed here instead.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.json")
    parser.add_argument("--annotation-root", default="annotation")
    parser.add_argument("--output", default="docs/annotation_density.json")
    return parser.parse_args()


def resolve(root: Path, name: str) -> Path:
    path = root / name
    return path if path.exists() else Path(name)


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    annotation_root = Path(args.annotation_root)

    grand = {
        "images": 0,
        "images_with_box": 0,
        "boxes_total": 0,
        "boxes_with_id": 0,
        "boxes_without_id": 0,
    }
    per_stream = {}
    hist_total: Counter = Counter()
    hist_with_id: Counter = Counter()

    for condition in config["conditions"]:
        for view_name, view_cfg in condition["views"].items():
            xml_path = resolve(annotation_root, view_cfg["annotation"])
            if not xml_path.exists():
                print(f"skip missing XML: {xml_path}")
                continue

            images = 0
            images_with_box = 0
            boxes_total = 0
            boxes_with_id = 0

            for image in ET.parse(xml_path).getroot().findall("image"):
                images += 1
                boxes = image.findall("box")
                n_total = len(boxes)
                n_with_id = 0
                for box in boxes:
                    attr = box.find("attribute[@name='id']")
                    if attr is not None and attr.text and attr.text.strip():
                        n_with_id += 1

                boxes_total += n_total
                boxes_with_id += n_with_id
                if n_total:
                    images_with_box += 1
                    hist_total[n_total] += 1
                if n_with_id:
                    hist_with_id[n_with_id] += 1

            key = f"{condition['name']}_{view_name}"
            per_stream[key] = {
                "images": images,
                "images_with_box": images_with_box,
                "boxes_total": boxes_total,
                "boxes_with_id": boxes_with_id,
                "boxes_without_id": boxes_total - boxes_with_id,
                "boxes_per_annotated_frame": boxes_total / images_with_box if images_with_box else 0.0,
            }
            grand["images"] += images
            grand["images_with_box"] += images_with_box
            grand["boxes_total"] += boxes_total
            grand["boxes_with_id"] += boxes_with_id
            grand["boxes_without_id"] += boxes_total - boxes_with_id

            print(
                f"{key:28s} images={images:6d} with_box={images_with_box:6d} "
                f"boxes={boxes_total:7d} with_id={boxes_with_id:7d} "
                f"no_id={boxes_total - boxes_with_id:6d} "
                f"density={per_stream[key]['boxes_per_annotated_frame']:.2f}"
            )

    density_total = grand["boxes_total"] / grand["images_with_box"] if grand["images_with_box"] else 0.0
    density_with_id = grand["boxes_with_id"] / grand["images_with_box"] if grand["images_with_box"] else 0.0

    print()
    print(f"images (all frames in XML)   : {grand['images']:,}")
    print(f"frames carrying >=1 box      : {grand['images_with_box']:,}")
    print(f"boxes, total in XML          : {grand['boxes_total']:,}")
    print(f"boxes carrying an identity   : {grand['boxes_with_id']:,}")
    print(f"boxes WITHOUT an identity    : {grand['boxes_without_id']:,}")
    print()
    print(f"density, all annotated boxes : {density_total:.2f} per annotated frame")
    print(f"density, id-carrying boxes   : {density_with_id:.2f} per annotated frame")
    print()
    if grand["boxes_without_id"]:
        share = grand["boxes_without_id"] / grand["boxes_total"] * 100
        print(
            f"NOTE: {share:.1f}% of annotated boxes have no identity and are absent from "
            "the crop manifest. Manifest-derived density understates annotation density; "
            "quote the all-boxes number above for any statement about how busy a frame is."
        )
    else:
        print(
            "Every annotated box carries an identity, so the manifest is complete and the "
            "manifest-derived density is the annotation density."
        )

    result = {"total": grand, "per_stream": per_stream,
              "boxes_per_annotated_frame_all": density_total,
              "boxes_per_annotated_frame_with_id": density_with_id,
              "histogram_boxes_per_frame_all": dict(hist_total),
              "histogram_boxes_per_frame_with_id": dict(hist_with_id)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
