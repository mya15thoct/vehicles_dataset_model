#!/usr/bin/env python3
"""Structural modules for WICV-Net v2.

Two components that exploit structure specific to this benchmark, rather than
adding another generic training-time metric constraint:

1. CrossViewTransition (CVT) -- the before->after view change in this dataset
   is *systematic*, not arbitrary: the same vehicle passes a fixed point and is
   observed front-first, rear-second. So the view gap is a learnable function,
   not just a distance to be minimized. CVT learns a directional residual map
   between the two view subspaces and, unlike every purely metric loss, stays
   active at inference time.

2. ConditionAdaptiveBNNeck (CAN) -- the loss-weight sweep showed that
   adversarially erasing the weather/time signal degrades accuracy. That is
   evidence the condition is not pure nuisance noise but a known covariate.
   CAN therefore models condition-specific feature statistics explicitly and
   normalizes them away, instead of fighting the backbone with a reversed
   gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

NUM_CONDITIONS = 4  # (time in {morning, evening}) x (weather in {norain, rain})


def condition_index(time_index: torch.Tensor, weather_index: torch.Tensor) -> torch.Tensor:
    """Flatten the 2x2 condition grid into a single index in [0, 4)."""
    return time_index * 2 + weather_index


class CrossViewTransition(nn.Module):
    """Directional residual maps between the before-view and after-view subspaces.

    Two small residual MLPs are learned: `b2a` maps a before-view embedding to
    where its after-view counterpart should live, and `a2b` maps the reverse
    direction. Outputs are L2-normalized so they live on the same unit sphere
    the retrieval metric operates on.

    At inference the gallery (before view) can be pushed through `b2a` so that
    query-gallery matching happens inside a single view subspace instead of
    across the raw view gap.
    """

    def __init__(self, feat_dim: int, hidden_ratio: float = 0.5, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(64, int(feat_dim * hidden_ratio))
        self.b2a = self._build_branch(feat_dim, hidden, dropout)
        self.a2b = self._build_branch(feat_dim, hidden, dropout)

    @staticmethod
    def _build_branch(feat_dim: int, hidden: int, dropout: float) -> nn.Module:
        layers = [nn.Linear(feat_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, feat_dim))
        return nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, direction: str) -> torch.Tensor:
        if direction == "b2a":
            residual = self.b2a(features)
        elif direction == "a2b":
            residual = self.a2b(features)
        else:
            raise ValueError(f"direction must be 'b2a' or 'a2b', got {direction!r}")
        return F.normalize(features + residual, p=2, dim=1)


class ConditionAdaptiveBNNeck(nn.Module):
    """BNNeck with one normalization branch per weather/time condition.

    Each of the four conditions keeps its own affine parameters and running
    statistics, so condition-specific first- and second-order feature shifts
    are removed before the identity head sees the embedding. A shared branch is
    also maintained and used whenever the condition label is unavailable, which
    keeps the model usable on unlabeled footage and makes the cross-condition
    protocol (where the test condition is held out) well defined.

    The condition label is scene metadata (timestamp plus weather), not a
    per-vehicle annotation, so using it at test time does not leak identity
    information. The shared fallback is what the paper should report whenever
    that assumption is not acceptable.
    """

    def __init__(self, feat_dim: int, num_conditions: int = NUM_CONDITIONS) -> None:
        super().__init__()
        self.num_conditions = num_conditions
        self.branches = nn.ModuleList(
            [self._build_bn(feat_dim) for _ in range(num_conditions)]
        )
        self.shared = self._build_bn(feat_dim)

    @staticmethod
    def _build_bn(feat_dim: int) -> nn.BatchNorm1d:
        bn = nn.BatchNorm1d(feat_dim)
        bn.bias.requires_grad_(False)  # standard BNNeck: bias is frozen
        return bn

    def forward(self, features: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is None:
            return self.shared(features)

        output = self.shared(features)
        for index in range(self.num_conditions):
            mask = condition == index
            if not mask.any():
                continue
            selected = features[mask]
            # BatchNorm1d needs >1 sample in train mode to estimate variance;
            # fall back to the shared branch for degenerate single-sample groups.
            if self.training and selected.shape[0] < 2:
                continue
            output = output.masked_scatter(
                mask.unsqueeze(1).expand_as(output),
                self.branches[index](selected),
            )
        return output
