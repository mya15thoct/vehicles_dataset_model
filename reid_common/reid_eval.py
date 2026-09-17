"""Shared crop dataset, feature extraction, and CMC/mAP retrieval metrics.

Used by baselines/osnet, baselines/torchreid, methods/wicv, and
conference/make_retrieval_figure.py so the evaluation protocol (transform,
normalization, ranking) is identical everywhere a model is scored.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

TIME_NAMES = ["morning", "evening"]
WEATHER_NAMES = ["norain", "rain"]


def condition_factors(condition: str) -> tuple[int, int]:
    """Split e.g. 'morning_rain' into (time_index, weather_index)."""
    parts = condition.split("_")
    if len(parts) != 2 or parts[0] not in TIME_NAMES or parts[1] not in WEATHER_NAMES:
        raise ValueError(f"Unrecognized condition name: {condition}")
    return TIME_NAMES.index(parts[0]), WEATHER_NAMES.index(parts[1])


def build_eval_transform(height: int = 256, width: int = 128):
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class CropDataset(Dataset):
    """Evaluation dataset returning (image, row_index[, condition_index]).

    Set `with_condition=True` for models with a condition-adaptive neck
    (e.g. WICV-Net); the condition index is scene metadata (time-of-day x
    weather) already present in the split CSV.
    """

    def __init__(self, rows: list[dict], transform, with_condition: bool = False) -> None:
        self.rows = rows
        self.transform = transform
        self.with_condition = with_condition

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        if not self.with_condition:
            return tensor, index
        time_index, weather_index = condition_factors(row["condition"])
        return tensor, index, time_index * 2 + weather_index


def extract_features(
    model,
    rows: list[dict],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    height: int = 256,
    width: int = 128,
    use_condition: bool = False,
    log_prefix: str = "eval",
) -> torch.Tensor:
    """Extract L2-normalized embeddings for a list of crop rows.

    Set `use_condition=True` for a model whose forward pass takes a
    `condition=` tensor (WICV-Net's condition-adaptive neck).
    """
    dataset = CropDataset(rows, build_eval_transform(height, width), with_condition=use_condition)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features = [None] * len(rows)
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if use_condition:
                images, indices, conditions = batch
                embeddings = model(images.to(device), condition=conditions.to(device))
            else:
                images, indices = batch
                embeddings = model(images.to(device))
            if isinstance(embeddings, (tuple, list)):
                embeddings = embeddings[-1]
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu()
            for offset, row_index in enumerate(indices.tolist()):
                features[row_index] = embeddings[offset]
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(
                    f"  {log_prefix} batch {batch_index}/{math.ceil(len(dataset) / batch_size)} "
                    f"images={min(batch_index * batch_size, len(dataset))}/{len(dataset)}",
                    flush=True,
                )
    return torch.stack(features, dim=0)


def compute_metrics_from_dist(dist: torch.Tensor, query_ids: list[str], gallery_ids: list[str]) -> dict:
    """CMC/mAP from a query-by-gallery distance matrix (smaller = closer)."""
    rank1 = 0
    rank5 = 0
    ap_sum = 0.0
    valid_queries = 0

    for index in range(dist.shape[0]):
        qid = query_ids[index]
        positives = [gid == qid for gid in gallery_ids]
        num_positives = sum(positives)
        if num_positives == 0:
            continue

        order = torch.argsort(dist[index]).tolist()
        ordered_matches = [positives[i] for i in order]

        valid_queries += 1
        rank1 += int(ordered_matches[0])
        rank5 += int(any(ordered_matches[:5]))

        hits = 0
        precision_sum = 0.0
        for rank, is_match in enumerate(ordered_matches, start=1):
            if is_match:
                hits += 1
                precision_sum += hits / rank
                if hits == num_positives:
                    break
        ap_sum += precision_sum / num_positives

    if valid_queries == 0:
        return {"valid_queries": 0, "rank1": 0.0, "rank5": 0.0, "mAP": 0.0}
    return {
        "valid_queries": valid_queries,
        "rank1": rank1 / valid_queries,
        "rank5": rank5 / valid_queries,
        "mAP": ap_sum / valid_queries,
    }


def compute_metrics(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_ids: list[str],
    gallery_ids: list[str],
) -> dict:
    scores = query_features @ gallery_features.t()
    return compute_metrics_from_dist(-scores, query_ids, gallery_ids)
