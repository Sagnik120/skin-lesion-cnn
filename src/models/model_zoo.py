"""
Model Zoo — 4 CNN architectures for skin lesion classification.

All models:
- Use ImageNet pretrained weights (timm library)
- Replace the classifier head with 7-class output
- Support a freeze_backbone_epochs phase for warm-up fine-tuning
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm


NUM_CLASSES = 7


# ── Base wrapper ──────────────────────────────────────────────────────────────

class SkinLesionModel(nn.Module):
    """Base class — wraps a timm backbone with a custom head."""

    def __init__(self, backbone_name: str, num_classes: int, dropout: float, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,          # remove original head
            global_pool="avg",
        )
        #in_features = self.backbone.num_features
        # Automatically determine actual backbone output size
        dummy_size = 300 if "efficientnet" in backbone_name else 224

        with torch.no_grad():
            dummy = torch.randn(1, 3, dummy_size, dummy_size)
            in_features = self.backbone(dummy).shape[1]
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("  Backbone frozen (only classifier head trains)")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("  Backbone unfrozen (full fine-tuning)")

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── EfficientNet-B3 ───────────────────────────────────────────────────────────

class EfficientNetModel(SkinLesionModel):
    """
    EfficientNet-B3 — best accuracy-to-parameter ratio.
    Input size: 300×300.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3, pretrained: bool = True):
        super().__init__("efficientnet_b3", num_classes, dropout, pretrained)
        self.model_name = "EfficientNet-B3"

    def get_last_conv_layer(self):
        """Used by Grad-CAM to hook the last convolutional layer."""
        return self.backbone.conv_head


# ── ResNet-50 ─────────────────────────────────────────────────────────────────

class ResNetModel(SkinLesionModel):
    """
    ResNet-50 — strong residual baseline.
    Input size: 224×224.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5, pretrained: bool = True):
        super().__init__("resnet50", num_classes, dropout, pretrained)
        self.model_name = "ResNet-50"

    def get_last_conv_layer(self):
        return self.backbone.layer4[-1].conv3


# ── DenseNet-121 ──────────────────────────────────────────────────────────────

class DenseNetModel(SkinLesionModel):
    """
    DenseNet-121 — dense connections, strong in medical imaging.
    Input size: 224×224.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4, pretrained: bool = True):
        super().__init__("densenet121", num_classes, dropout, pretrained)
        self.model_name = "DenseNet-121"

    def get_last_conv_layer(self):
        return self.backbone.features.denseblock4.denselayer16.conv2


# ── MobileNet-V3 ──────────────────────────────────────────────────────────────

class MobileNetModel(SkinLesionModel):
    """
    MobileNet-V3 Large — lightweight, deployment-ready.
    Input size: 224×224.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.3, pretrained: bool = True):
        super().__init__("mobilenetv3_large_100", num_classes, dropout, pretrained)
        self.model_name = "MobileNet-V3"

    def get_last_conv_layer(self):
        return self.backbone.conv_head


# ── Factory ───────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "efficientnet": EfficientNetModel,
    "resnet":       ResNetModel,
    "densenet":     DenseNetModel,
    "mobilenet":    MobileNetModel,
}


def build_model(model_name: str, config: dict) -> SkinLesionModel:
    """
    Build a model from config dict.

    Args:
        model_name: one of 'efficientnet', 'resnet', 'densenet', 'mobilenet'
        config: dict with keys: num_classes, dropout, pretrained

    Returns:
        Initialized SkinLesionModel
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(
        num_classes=config.get("num_classes", NUM_CLASSES),
        dropout=config.get("dropout", 0.3),
        pretrained=config.get("pretrained", True),
    )

    params = model.count_params()
    print(f"Built {model.model_name}")
    print(f"  Total params   : {params['total']:,}")
    print(f"  Trainable params: {params['trainable']:,}")

    return model


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}\n")

    for name in MODEL_REGISTRY:
        model = build_model(name, {"num_classes": 7, "dropout": 0.3, "pretrained": False})
        model.to(device)
        dummy_size = 300 if name == "efficientnet" else 224
        x = torch.randn(2, 3, dummy_size, dummy_size, device=device)
        out = model(x)
        assert out.shape == (2, 7), f"Unexpected output shape: {out.shape}"
        print(f"  Forward pass OK → output: {out.shape}\n")

    print("All models passed ✓")
