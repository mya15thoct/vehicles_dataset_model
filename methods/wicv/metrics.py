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
    use_condition: bool | None = None,
) -> torch.Tensor:
    """Extract L2-normalized embeddings.

    `use_condition` routes each sample through its condition branch when the
    model has a condition-adaptive neck; it defaults to whatever the model was
    built with. Pass False to force the shared-branch fallback, which is what
    the cross-condition protocol needs when the test condition is unseen.
    """
    if use_condition is None:
        use_condition = bool(getattr(model, "use_can", False))

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
        for batch_index, (images, indices, conditions) in enumerate(loader, start=1):
            condition = conditions.to(device) if use_condition else None
            embeddings = model(images.to(device), condition=condition).cpu()
            for offset, row_index in enumerate(indices.tolist()):
                features[row_index] = embeddings[offset]
            if batch_index % 20 == 0 or batch_index == len(loader):
                print(
                    f"  eval batch {batch_index}/{math.ceil(len(dataset) / batch_size)} "
                    f"images={min(batch_index * batch_size, len(dataset))}/{len(dataset)}",
                    flush=True,
                )
    return torch.stack(features, dim=0)


def apply_cross_view_transition(
    model,
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    device: torch.device,
    mode: str = "gallery",
    batch_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map features through the learned view transition before matching.

    mode='gallery' pushes the before-view gallery into the after-view subspace
    (the query's own space); mode='query' does the reverse. Returns the pair of
    feature matrices to score against each other.
    """
    if getattr(model, "transition", None) is None:
        return query_features, gallery_features

    def transform(features: torch.Tensor, direction: str) -> torch.Tensor:
        chunks = []
        model.eval()
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                chunk = features[start:start + batch_size].to(device)
                chunks.append(model.transform(chunk, direction).cpu())
        return torch.cat(chunks, dim=0)

    if mode == "gallery":
        return query_features, transform(gallery_features, "b2a")
    if mode == "query":
        return transform(query_features, "a2b"), gallery_features
    raise ValueError(f"mode must be 'gallery' or 'query', got {mode!r}")


def compute_metrics(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_ids: list[str],
    gallery_ids: list[str],
) -> dict:
    scores = query_features @ gallery_features.t()
    return compute_metrics_from_dist(-scores, query_ids, gallery_ids)


def compute_metrics_from_dist(
    dist: torch.Tensor,
    query_ids: list[str],
    gallery_ids: list[str],
) -> dict:
    """Metrics from a query-by-gallery distance matrix (smaller = closer)."""
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


def evaluate_retrieval(
    model,
    query_rows: list[dict],
    gallery_rows: list[dict],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    height: int,
    width: int,
    cvt_mode: str = "gallery",
) -> dict:
    """Validation-time retrieval.

    When the model carries a transition module the same mapping used at test
    time is applied here too, so validation-mAP model selection optimizes the
    procedure that will actually be reported.
    """
    query_features = extract_features(model, query_rows, batch_size, num_workers, device, height, width)
    gallery_features = extract_features(model, gallery_rows, batch_size, num_workers, device, height, width)
    if getattr(model, "transition", None) is not None and cvt_mode != "off":
        query_features, gallery_features = apply_cross_view_transition(
            model, query_features, gallery_features, device, mode=cvt_mode
        )
    return compute_metrics(
        query_features,
        gallery_features,
        [identity(row) for row in query_rows],
        [identity(row) for row in gallery_rows],
    )
