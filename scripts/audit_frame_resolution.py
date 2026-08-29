#!/usr/bin/env python3
"""Check that every annotated frame matches the size the XML declares for it.

Frame resolution is not uniform: some streams mix 1080x1920 and 1440x2560
frames. CVAT records width and height per image, so box coordinates are only
meaningful against the size declared for that frame. If a file on disk has a
different size from its XML entry, the coordinates refer to a different pixel
grid, and the crop exporter will silently cut the wrong region -- it clamps to
the real image bounds without ever comparing the two.

Reports, per stream: the resolutions present, how many frames disagree with
their XML entry, and examples. A clean run means the mixed resolutions are only
a documentation issue; any mismatch means crops have to be re-exported.
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.json")
    parser.add_argument("--image-root", default="/mnt/ngan/vehicles/multi-weather_traffic_data")
    parser.add_argument("--annotation-root", default="annotation")
    parser.add_argument("--output", default="docs/frame_resolution_audit.json")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Check only the first N frames per stream (0 = all). Use a small "
        "value for a quick look; the full pass reads every file header.",
    )
    return parser.parse_args()


def resolve(root: Path, name: str) -> Path:
    path = root / name
    return path if path.exists() else Path(name)


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    annotation_root = Path(args.annotation_root)
    image_root = Path(args.image_root)

    report = {}
    total_mismatch = 0
    total_missing = 0
    total_checked = 0

    for condition in config["conditions"]:
        for view_name, view_cfg in condition["views"].items():
            xml_path = resolve(annotation_root, view_cfg["annotation"])
            image_dir = resolve(image_root, view_cfg["images"])
            stream = f"{condition['name']}_{view_name}"

            if not xml_path.exists():
                print(f"skip missing XML: {xml_path}")
                continue
            if not image_dir.exists():
                print(f"skip missing images: {image_dir}")
                continue

            declared = Counter()
            actual = Counter()
            mismatches = []
            missing = 0
            checked = 0

            images = ET.parse(xml_path).getroot().findall("image")
            if args.limit:
                images = images[: args.limit]

            for image in images:
                name = image.attrib["name"]
                dw = int(image.attrib.get("width", 0))
                dh = int(image.attrib.get("height", 0))
                declared[(dw, dh)] += 1

                path = image_dir / name
                if not path.exists():
                    missing += 1
                    continue
                with Image.open(path) as source:  # header only, no decode
                    aw, ah = source.size
                actual[(aw, ah)] += 1
                checked += 1
                if (dw, dh) != (aw, ah):
                    if len(mismatches) < 5:
                        mismatches.append(
                            {"frame": name, "xml": [dw, dh], "file": [aw, ah],
                             "boxes": len(image.findall("box"))}
                        )
                    total_mismatch += 1

            total_missing += missing
            total_checked += checked
            report[stream] = {
                "frames_checked": checked,
                "frames_missing_on_disk": missing,
                "declared_sizes": {f"{w}x{h}": n for (w, h), n in sorted(declared.items())},
                "actual_sizes": {f"{w}x{h}": n for (w, h), n in sorted(actual.items())},
                "mismatch_count": sum(1 for _ in mismatches) if len(mismatches) < 5 else "5+",
                "mismatch_examples": mismatches,
            }

            sizes = " ".join(f"{w}x{h}:{n}" for (w, h), n in sorted(actual.items()))
            flag = "  <-- MISMATCH" if mismatches else ""
            print(f"{stream:26s} checked={checked:6d} missing={missing:5d}  {sizes}{flag}")

    print()
    print(f"frames checked        : {total_checked:,}")
    print(f"frames missing on disk: {total_missing:,}")
    print(f"size mismatches       : {total_mismatch:,}")
    print()
    if total_mismatch:
        print("A mismatch means the box coordinates for that frame were drawn against a")
        print("different pixel grid from the file on disk, so its crops were cut from the")
        print("wrong region. Those streams need the crops re-exported after the frames and")
        print("the XML are brought back into agreement.")
    else:
        print("Every frame matches its XML entry, so the mixed resolutions are a")
        print("documentation matter only: coordinates are correct against their own frame,")
        print("and the exported crops are valid. Report both resolutions in the paper.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {output}")
    return 1 if total_mismatch else 0


if __name__ == "__main__":
    raise SystemExit(main())
