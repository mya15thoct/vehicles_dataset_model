#!/usr/bin/env python3
"""Dataset and batch sampler for WICV-Net training on Re-ID crop CSV splits."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, Sampler

TIME_NAMES = ["morning", "evening"]
WEATHER_NAMES = ["norain", "rain"]
VIEW_NAMES = ["before", "after"]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def identity(row: dict) -> str:
    return f"{row['condition']}::{int(row['vehicle_id']):06d}"


def normalize_view(view: str) -> str:
    if view.startswith("before"):
        return "before"
    if view.startswith("after"):
        return "after"
    return view


def condition_factors(condition: str) -> tuple[int, int]:
    """Split e.g. 'morning_rain' into (time_index, weather_index)."""
    parts = condition.split("_")
    if len(parts) != 2 or parts[0] not in TIME_NAMES or parts[1] not in WEATHER_NAMES:
        raise ValueError(f"Unrecognized condition name: {condition}")
    return TIME_NAMES.index(parts[0]), WEATHER_NAMES.index(parts[1])


class ReidTrainDataset(Dataset):
    """Returns (image, identity_index, view_index, time_index, weather_index)."""

    def __init__(self, rows: list[dict], label_to_index: dict[str, int], transform) -> None:
        self.rows = rows
        self.label_to_index = label_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        label = self.label_to_index[identity(row)]
        view = VIEW_NAMES.index(normalize_view(row["view"]))
        time_index, weather_index = condition_factors(row["condition"])
        return tensor, label, view, time_index, weather_index


class CropDataset(Dataset):
    """Evaluation dataset returning (image, row_index)."""

    def __init__(self, rows: list[dict], transform) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, index


class CrossViewIdentitySampler(Sampler):
    """PK sampler that balances the two camera views inside each identity group.

    Each batch contains `num_ids` identities with `num_instances` crops per
    identity. When an identity has crops in both views, half of its instances
    are drawn from `before` and half from `after`, which guarantees that the
    cross-view triplet and prototype losses always receive cross-view
    positives.
    """

    def __init__(self, rows: list[dict], label_to_index: dict[str, int], num_ids: int, num_instances: int, seed: int = 42) -> None:
        self.num_ids = num_ids
        self.num_instances = num_instances
        self.rng = random.Random(seed)
        self.index_by_id_view: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(rows):
            label = label_to_index[identity(row)]
            self.index_by_id_view[label][normalize_view(row["view"])].append(index)
        self.labels = sorted(self.index_by_id_view.keys())
        self.batches_per_epoch = max(1, len(self.labels) // self.num_ids)

    def _sample_for_identity(self, label: int) -> list[int]:
        by_view = self.index_by_id_view[label]
        views = [view for view in VIEW_NAMES if by_view.get(view)]
        picked: list[int] = []
        if len(views) == 2:
            half = self.num_instances // 2
            counts = {"before": half, "after": self.num_instances - half}
            for view, count in counts.items():
                pool = by_view[view]
                if len(pool) >= count:
                    picked.extend(self.rng.sample(pool, count))
                else:
                    picked.extend(self.rng.choices(pool, k=count))
        else:
            pool = by_view[views[0]]
            if len(pool) >= self.num_instances:
                picked.extend(self.rng.sample(pool, self.num_instances))
            else:
                picked.extend(self.rng.choices(pool, k=self.num_instances))
        return picked

    def __iter__(self):
        labels = self.labels[:]
        self.rng.shuffle(labels)
        for batch_index in range(self.batches_per_epoch):
            batch_labels = labels[batch_index * self.num_ids:(batch_index + 1) * self.num_ids]
            batch: list[int] = []
            for label in batch_labels:
                batch.extend(self._sample_for_identity(label))
            yield from batch

    def __len__(self) -> int:
        return self.batches_per_epoch * self.num_ids * self.num_instances
