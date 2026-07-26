#!/usr/bin/env python3
"""Loss functions for WICV-Net.

Components:
1. cross_view_batch_hard_triplet: batch-hard triplet whose hardest positive is
   constrained to the opposite camera view when one exists, directly
   optimizing the after->before retrieval objective of the benchmark.
2. CrossViewPrototypeMemory: EMA memory of one prototype per (identity, view);
   every embedding is contrastively pulled toward the *opposite-view*
   prototype of its own identity (cross-view prototype alignment, CVPA).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def cross_view_batch_hard_triplet(
    features: torch.Tensor,
    labels: torch.Tensor,
    views: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Batch-hard triplet loss with cross-view positive mining.

    For each anchor the hardest positive is searched among samples of the same
    identity captured by the *other* camera view. Anchors whose identity has no
    cross-view sample in the batch fall back to standard batch-hard positives.
    """
    dist = torch.cdist(features, features, p=2)
    batch_size = features.shape[0]
    same_id = labels.unsqueeze(0) == labels.unsqueeze(1)
    not_self = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
    pos_mask = same_id & not_self
    cross_view = views.unsqueeze(0) != views.unsqueeze(1)
    cv_pos_mask = pos_mask & cross_view

    has_pos = pos_mask.any(dim=1)
    has_cv_pos = cv_pos_mask.any(dim=1)
    effective_pos = torch.where(has_cv_pos.unsqueeze(1), cv_pos_mask, pos_mask)

    neg_mask = ~same_id
    hardest_pos = dist.masked_fill(~effective_pos, float("-inf")).max(dim=1).values
    hardest_neg = dist.masked_fill(~neg_mask, float("inf")).min(dim=1).values

    valid = has_pos & neg_mask.any(dim=1)
    if not valid.any():
        return features.new_zeros(())
    loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return loss.mean()


def cross_view_transition_loss(
    transition: nn.Module,
    bn_features: torch.Tensor,
    labels: torch.Tensor,
    views: torch.Tensor,
) -> torch.Tensor:
    """Train the directional view maps on in-batch cross-view positive pairs.

    For every (before, after) pair of the same identity present in the batch,
    `b2a` must carry the before embedding onto its after counterpart and `a2b`
    the reverse. The *target* of each direction is detached: CVT's job is to
    learn the transformation between the two view subspaces, not to drag them
    together -- that is CVPA's job, and letting both gradients flow would let
    the two objectives satisfy each other by collapsing the view distinction.

    The view-balanced PK sampler guarantees such pairs exist in a batch
    whenever an identity is observed in both views.
    """
    same_id = labels.unsqueeze(0) == labels.unsqueeze(1)
    before_to_after = same_id & (views.unsqueeze(1) == 0) & (views.unsqueeze(0) == 1)
    after_to_before = same_id & (views.unsqueeze(1) == 1) & (views.unsqueeze(0) == 0)

    normalized = F.normalize(bn_features, p=2, dim=1)
    total = bn_features.new_zeros(())
    terms = 0

    for mask, direction in ((before_to_after, "b2a"), (after_to_before, "a2b")):
        source_index, target_index = mask.nonzero(as_tuple=True)
        if source_index.numel() == 0:
            continue
        mapped = transition(bn_features[source_index], direction)
        target = normalized[target_index].detach()
        total = total + (1.0 - (mapped * target).sum(dim=1)).mean()
        terms += 1

    return total / terms if terms else total


class CrossViewPrototypeMemory(nn.Module):
    """EMA prototype memory for cross-view prototype alignment (CVPA).

    Keeps one L2-normalized prototype per (identity, view). The loss is an
    InfoNCE classification of each embedding against the prototypes of the
    opposite view, so the representation of a vehicle seen from the rear is
    explicitly anchored to its front-view prototype and vice versa.
    """

    def __init__(self, num_ids: int, feat_dim: int, momentum: float = 0.9, temperature: float = 0.07) -> None:
        super().__init__()
        self.momentum = momentum
        self.temperature = temperature
        self.register_buffer("prototypes", torch.zeros(num_ids, 2, feat_dim))
        self.register_buffer("valid", torch.zeros(num_ids, 2, dtype=torch.bool))

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor, views: torch.Tensor) -> None:
        features = F.normalize(features, p=2, dim=1)
        for feat, label, view in zip(features, labels.tolist(), views.tolist()):
            if self.valid[label, view]:
                updated = self.momentum * self.prototypes[label, view] + (1.0 - self.momentum) * feat
            else:
                updated = feat
                self.valid[label, view] = True
            self.prototypes[label, view] = F.normalize(updated, p=2, dim=0)

    def loss(self, features: torch.Tensor, labels: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, p=2, dim=1)
        opposite = 1 - views
        target_valid = self.valid[labels, opposite]
        if not target_valid.any():
            return features.new_zeros(())

        features = features[target_valid]
        labels = labels[target_valid]
        opposite = opposite[target_valid]

        protos = F.normalize(self.prototypes, p=2, dim=2)
        sim_before = features @ protos[:, 0].t()
        sim_after = features @ protos[:, 1].t()
        logits = torch.where((opposite == 0).unsqueeze(1), sim_before, sim_after)

        neg_valid = torch.where(
            (opposite == 0).unsqueeze(1),
            self.valid[:, 0].unsqueeze(0),
            self.valid[:, 1].unsqueeze(0),
        )
        logits = logits.masked_fill(~neg_valid, -1e4) / self.temperature
        return F.cross_entropy(logits, labels)
