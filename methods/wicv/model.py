#!/usr/bin/env python3
"""WICV-Net model: Torchreid backbone + BNNeck + factorized condition-adversarial heads."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from modules import ConditionAdaptiveBNNeck, CrossViewTransition


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


TORCHVISION_BACKBONES = ("tv_swin_t", "tv_swin_s", "tv_vit_b_16", "tv_convnext_tiny")


def build_torchvision_backbone(model_name: str, pretrained: bool) -> nn.Module:
    from torchvision import models

    if model_name == "tv_swin_t":
        backbone = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.head = nn.Identity()
    elif model_name == "tv_swin_s":
        backbone = models.swin_s(weights=models.Swin_S_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.head = nn.Identity()
    elif model_name == "tv_vit_b_16":
        backbone = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone.heads = nn.Identity()
    elif model_name == "tv_convnext_tiny":
        backbone = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone.classifier[2] = nn.Identity()
    else:
        raise SystemExit(f"Unknown torchvision backbone: {model_name}")
    return backbone


def build_backbone(model_name: str, pretrained: bool):
    if model_name in TORCHVISION_BACKBONES:
        return build_torchvision_backbone(model_name, pretrained)

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
    """Backbone + (condition-adaptive) BNNeck + optional CVT / adversarial heads.

    forward() in train mode returns a dict with:
      features     -- raw backbone embedding used by triplet / prototype losses
      bn_features  -- BNNeck embedding; the space retrieval actually runs in,
                      and therefore the space CVT is trained on
      id_logits    -- identity logits from the BNNeck feature
      time_logits / weather_logits -- gradient-reversed condition logits (FCA)
    In eval mode it returns the L2-normalized BNNeck embedding.

    `use_can` swaps the single shared BNNeck for a condition-adaptive one, and
    `use_cvt` attaches the directional cross-view transition module; both are
    off by default so the v1 configuration is reproducible unchanged.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        height: int = 256,
        width: int = 128,
        use_cvt: bool = False,
        use_can: bool = False,
        cvt_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_name == "tv_vit_b_16" and (height, width) != (224, 224):
            raise SystemExit("tv_vit_b_16 requires --height 224 --width 224 (fixed positional embeddings).")
        if model_name.startswith("tv_swin") and (height % 32 or width % 32):
            raise SystemExit("Swin backbones require height/width divisible by 32.")
        self.model_name = model_name
        self.use_cvt = use_cvt
        self.use_can = use_can
        self.backbone = build_backbone(model_name, pretrained)
        self.feat_dim = infer_feature_dim(self.backbone, height, width)

        if use_can:
            self.bnneck = ConditionAdaptiveBNNeck(self.feat_dim)
        else:
            self.bnneck = nn.BatchNorm1d(self.feat_dim)
            self.bnneck.bias.requires_grad_(False)
        self.id_classifier = nn.Linear(self.feat_dim, num_classes, bias=False)
        self.transition = CrossViewTransition(self.feat_dim, dropout=cvt_dropout) if use_cvt else None

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

    def _neck(self, features: torch.Tensor, condition: torch.Tensor | None) -> torch.Tensor:
        if self.use_can:
            return self.bnneck(features, condition)
        return self.bnneck(features)

    def transform(self, features: torch.Tensor, direction: str) -> torch.Tensor:
        """Apply the learned cross-view map to already-extracted embeddings."""
        if self.transition is None:
            raise RuntimeError("This checkpoint was trained without CVT (use_cvt=False).")
        return self.transition(features, direction)

    def forward(
        self,
        images: torch.Tensor,
        grl_weight: float = 1.0,
        condition: torch.Tensor | None = None,
    ):
        features = self._backbone_features(images)
        bn_features = self._neck(features, condition)
        if not self.training:
            return F.normalize(bn_features, p=2, dim=1)

        reversed_features = grad_reverse(bn_features, grl_weight)
        return {
            "features": features,
            "bn_features": bn_features,
            "id_logits": self.id_classifier(bn_features),
            "time_logits": self.time_head(reversed_features),
            "weather_logits": self.weather_head(reversed_features),
        }
