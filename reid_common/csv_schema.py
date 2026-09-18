"""Crop-manifest/split CSV schema and identity helpers shared by scripts/ and conference/."""

from __future__ import annotations

import csv
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


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to a crop-manifest-schema CSV, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in FIELDS})


def identity(row: dict) -> str:
    return f"{row['condition']}::{int(row['vehicle_id']):06d}"


def normalize_view(view: str) -> str:
    if view.startswith("before"):
        return "before"
    if view.startswith("after"):
        return "after"
    return view
