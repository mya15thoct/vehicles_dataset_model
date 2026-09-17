#!/usr/bin/env python3
"""Feature extraction and retrieval metrics shared by WICV-Net train/eval."""

from __future__ import annotations

import torch

from reid_common.reid_eval import compute_metrics, compute_metrics_from_dist, extract_features
from dataset import identity


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
    use_condition = bool(getattr(model, "use_can", False))
    query_features = extract_features(model, query_rows, batch_size, num_workers, device, height, width, use_condition)
    gallery_features = extract_features(model, gallery_rows, batch_size, num_workers, device, height, width, use_condition)
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
