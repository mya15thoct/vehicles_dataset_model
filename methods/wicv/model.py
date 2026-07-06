#!/usr/bin/env python3
"""WICV-Net model: Torchreid backbone + BNNeck + factorized condition-adversarial heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, weight: float) -> torch.Tensor:
        ctx.weight = weight
        return tensor.view_as(tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.weight * grad_output, None


def grad_reverse(tensor: torch.Tensor, weight: float) -> torch.Tensor:
    return GradientReversal.apply(tensor, weight)


def build_backbone(model_name: str, pretrained: bool):
    try:
        import torchreid
    except ImportError as exc:
        raise SystemExit("Missing torchreid. Install torchreid and its dependencies first.") from exc

    # loss='triplet' makes torchreid models return (logits, features) in train
    # mode; the internal classifier is unused because WICV-Net owns its heads.
    backbone = torchreid.models.build_model(
        name=model_name,
        num_classes=1,
        pretrained=pretrained,
        loss="triplet",
    )
    return backbone


def infer_feature_dim(backbone: nn.Module, height: int, width: int) -> int:
    backbone.eval()
    with torch.no_grad():
        features = backbone(torch.zeros(1, 3, height, width))
    if isinstance(features, (tuple, list)):
        features = features[-1]
    return features.shape[1]


class WICVNet(nn.Module):
    """Backbone + BNNeck identity head + gradient-reversed time/weather heads.

    forward() in train mode returns a dict with:
      features    -- raw embedding used by triplet / prototype losses
      id_logits   -- identity logits from the BNNeck feature
      time_logits -- 2-way (morning/evening) adversarial logits
      weather_logits -- 2-way (norain/rain) adversarial logits
    In eval mode it returns the L2-normalized BNNeck embedding.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        height: int = 256,
        width: int = 128,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.backbone = build_backbone(model_name, pretrained)
        self.feat_dim = infer_feature_dim(self.backbone, height, width)

        self.bnneck = nn.BatchNorm1d(self.feat_dim)
        self.bnneck.bias.requires_grad_(False)
        self.id_classifier = nn.Linear(self.feat_dim, num_classes, bias=False)

        self.time_head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim // 4, 2),
        )
        self.weather_head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim // 4, 2),
        )

    def _backbone_features(self, images: torch.Tensor) -> torch.Tensor:
        output = self.backbone(images)
        if isinstance(output, (tuple, list)):
            output = output[-1]
        return output

    def forward(self, images: torch.Tensor, grl_weight: float = 1.0):
        features = self._backbone_features(images)
        bn_features = self.bnneck(features)
        if not self.training:
            return F.normalize(bn_features, p=2, dim=1)

        reversed_features = grad_reverse(bn_features, grl_weight)
        return {
            "features": features,
            "id_logits": self.id_classifier(bn_features),
            "time_logits": self.time_head(reversed_features),
            "weather_logits": self.weather_head(reversed_features),
        }
