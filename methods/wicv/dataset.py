#!/usr/bin/env python3
"""Dataset and batch sampler for WICV-Net training on Re-ID crop CSV splits."""

from __future__ import annotations

import random
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset, Sampler

from reid_common.csv_schema import identity, normalize_view, read_csv
from reid_common.reid_eval import condition_factors

VIEW_NAMES = ["before", "after"]


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
