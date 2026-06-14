"""
Dual-input EfficientNet-B0 model for neonatal jaundice detection.

The model has two image branches:
1. Skin branch: learns color/texture features from skin-region photos.
2. Sclera branch: learns yellowing cues from the eye-white region.

The fused representation feeds two heads:
- A 4-class severity classifier.
- A bilirubin regression head that estimates mg/dL.

Clinical caution:
Regression output is a model estimate and must not be used as a replacement for
serum bilirubin validation in a production medical workflow.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torchvision import models

import config


class EfficientNetFeatureBranch(nn.Module):
    """EfficientNet-B0 feature extractor plus a compact projection layer."""

    def __init__(self, pretrained: bool = config.PRETRAINED, projection_dim: int = 256) -> None:
        super().__init__()

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # EfficientNet-B0 classifier is Dropout + Linear. We keep the feature
        # extractor and replace the classifier with Identity.
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.projection = nn.Sequential(
            nn.Linear(in_features, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        return self.projection(features)


class DualInputJaundiceNet(nn.Module):
    """
    Multi-task dual-input network for classification and bilirubin regression.

    Args:
        num_classes: Number of severity classes.
        pretrained: Use ImageNet pretrained EfficientNet-B0 weights.
        dropout: Dropout probability for fusion and task heads.
        ycbcr_feature_dim: Optional handcrafted YCbCr feature vector length.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = config.PRETRAINED,
        dropout: float = config.DROPOUT_RATE,
        ycbcr_feature_dim: int = config.YCBCR_FEATURE_DIM,
    ) -> None:
        super().__init__()

        branch_dim = 256
        self.skin_branch = EfficientNetFeatureBranch(pretrained=pretrained, projection_dim=branch_dim)
        self.sclera_branch = EfficientNetFeatureBranch(pretrained=pretrained, projection_dim=branch_dim)
        self.ycbcr_feature_dim = ycbcr_feature_dim

        fusion_input_dim = (branch_dim * 2) + ycbcr_feature_dim
        fusion_dim = 256

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.classification_head = nn.Linear(fusion_dim // 2, num_classes)

        # Softplus keeps bilirubin estimates non-negative while allowing smooth
        # gradients. Output shape is [batch].
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim // 2, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        skin_image: torch.Tensor,
        sclera_image: torch.Tensor,
        ycbcr_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        skin_features = self.skin_branch(skin_image)
        sclera_features = self.sclera_branch(sclera_image)

        if ycbcr_features is None:
            ycbcr_features = torch.zeros(
                skin_features.size(0),
                self.ycbcr_feature_dim,
                device=skin_features.device,
                dtype=skin_features.dtype,
            )
        else:
            ycbcr_features = ycbcr_features.to(device=skin_features.device, dtype=skin_features.dtype)

        fused = torch.cat([skin_features, sclera_features, ycbcr_features], dim=1)
        fused = self.fusion(fused)

        logits = self.classification_head(fused)
        bilirubin = self.regression_head(fused).squeeze(1)

        return {
            "logits": logits,
            "bilirubin": bilirubin,
        }


def build_model(device: Optional[torch.device] = None) -> DualInputJaundiceNet:
    """Factory used by training, evaluation, and the future Flask app."""

    model = DualInputJaundiceNet()
    if device is not None:
        model = model.to(device)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters for reports/debugging."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    net = build_model()
    print(net.__class__.__name__)
    print(f"Trainable parameters: {count_trainable_parameters(net):,}")
