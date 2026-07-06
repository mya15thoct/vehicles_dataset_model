#!/usr/bin/env python3
"""Feature extraction and retrieval metrics shared by WICV-Net train/eval."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CropDataset, identity


def build_eval_transform(height: int, width: int):
    return transforms.Compose(
        [
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def extract_features(
    model,
    rows: list[dict],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    height: int,
    width: int,
) -> torch.Tensor:
    dataset = CropDataset(rows, build_eval_transform(height, width))
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
        for batch_index, (images, indices) in enumerate(loader, start=1):
            embeddings = model(images.to(device)).cpu()
            for offset, row_index in enumerate(indices.tolist()):
                features[row_index] = embeddings[offset]
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(
                    f"  eval batch {batch_index}/{math.ceil(len(dataset) / batch_size)} "
                    f"images={min(batch_index * batch_size, len(dataset))}/{len(dataset)}",
                    flush=True,
                )
    return torch.stack(features, dim=0)


def compute_metrics(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_ids: list[str],
    gallery_ids: list[str],
) -> dict:
    rank1 = 0
    rank5 = 0
    ap_sum = 0.0
    valid_queries = 0

    for index in range(query_features.shape[0]):
        qid = query_ids[index]
        positives = [gid == qid for gid in gallery_ids]
        num_positives = sum(positives)
        if num_positives == 0:
            continue

        scores = torch.mv(gallery_features, query_features[index])
        order = torch.argsort(scores, descending=True).tolist()
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


def evaluate_retrieval(
    model,
    query_rows: list[dict],
    gallery_rows: list[dict],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    height: int,
    width: int,
) -> dict:
    query_features = extract_features(model, query_rows, batch_size, num_workers, device, height, width)
    gallery_features = extract_features(model, gallery_rows, batch_size, num_workers, device, height, width)
    return compute_metrics(
        query_features,
        gallery_features,
        [identity(row) for row in query_rows],
        [identity(row) for row in gallery_rows],
    )
