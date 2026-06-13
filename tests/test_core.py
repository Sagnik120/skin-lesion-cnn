"""
Unit tests — run with: python -m pytest tests/ -v
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data.dataset import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS
from src.models.model_zoo import build_model, MODEL_REGISTRY
from src.training.losses import FocalLoss


# ── Dataset tests ──────────────────────────────────────────────────────────────

class TestClassMappings:
    def test_class_names_count(self):
        assert len(CLASS_NAMES) == 7

    def test_class_to_idx_roundtrip(self):
        for cls in CLASS_NAMES:
            idx = CLASS_TO_IDX[cls]
            assert IDX_TO_CLASS[idx] == cls

    def test_no_duplicate_indices(self):
        indices = list(CLASS_TO_IDX.values())
        assert len(set(indices)) == len(indices)


# ── Model tests ────────────────────────────────────────────────────────────────

class TestModels:
    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_forward_pass(self, model_name):
        """Each model should accept a 3-channel image and output 7 logits."""
        model = build_model(model_name, {"num_classes": 7, "dropout": 0.0, "pretrained": False})
        model.eval()
        dummy_size = 300 if model_name == "efficientnet" else 224
        x = torch.randn(2, 3, dummy_size, dummy_size)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 7), f"{model_name}: expected (2,7), got {out.shape}"

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_freeze_unfreeze(self, model_name):
        model = build_model(model_name, {"num_classes": 7, "dropout": 0.0, "pretrained": False})
        model.freeze_backbone()
        assert not any(p.requires_grad for p in model.backbone.parameters())
        model.unfreeze_backbone()
        assert all(p.requires_grad for p in model.backbone.parameters())

    @pytest.mark.parametrize("model_name", list(MODEL_REGISTRY.keys()))
    def test_count_params(self, model_name):
        model = build_model(model_name, {"num_classes": 7, "dropout": 0.0, "pretrained": False})
        params = model.count_params()
        assert params["total"] > 0
        assert params["trainable"] <= params["total"]


# ── Focal loss tests ───────────────────────────────────────────────────────────

class TestFocalLoss:
    def test_output_shape(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits  = torch.randn(8, 7)
        labels  = torch.randint(0, 7, (8,))
        loss    = loss_fn(logits, labels)
        assert loss.shape == ()  # scalar

    def test_focal_less_than_ce(self):
        """Focal loss should be <= cross-entropy for confident predictions."""
        import torch.nn as nn
        logits = torch.zeros(4, 7)
        # Make the model very confident on class 0
        logits[:, 0] = 10.0
        labels = torch.zeros(4, dtype=torch.long)

        fl = FocalLoss(gamma=2.0)(logits, labels).item()
        ce = nn.CrossEntropyLoss()(logits, labels).item()
        assert fl <= ce + 1e-5, f"Focal {fl:.4f} should be <= CE {ce:.4f} for easy examples"

    def test_with_class_weights(self):
        weights = torch.ones(7) * 0.5
        loss_fn = FocalLoss(alpha=weights, gamma=2.0)
        logits  = torch.randn(4, 7)
        labels  = torch.randint(0, 7, (4,))
        loss    = loss_fn(logits, labels)
        assert torch.isfinite(loss)

    def test_no_nan(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits  = torch.randn(16, 7)
        labels  = torch.randint(0, 7, (16,))
        loss    = loss_fn(logits, labels)
        assert not torch.isnan(loss)


# ── Metrics sanity tests ───────────────────────────────────────────────────────

class TestMetrics:
    def test_compute_metrics_perfect(self):
        from src.evaluation.evaluator import compute_metrics
        n = 70  # 10 per class × 7
        labels = np.repeat(np.arange(7), 10)
        preds  = labels.copy()
        probs  = np.eye(7)[labels]
        metrics = compute_metrics(labels, preds, probs)
        assert abs(metrics["accuracy"] - 1.0) < 1e-6
        assert abs(metrics["f1_macro"] - 1.0)  < 1e-6

    def test_compute_metrics_keys(self):
        from src.evaluation.evaluator import compute_metrics
        labels = np.random.randint(0, 7, 100)
        preds  = np.random.randint(0, 7, 100)
        probs  = np.random.dirichlet(np.ones(7), size=100)
        metrics = compute_metrics(labels, preds, probs)
        assert "accuracy"      in metrics
        assert "f1_macro"      in metrics
        assert "roc_auc_macro" in metrics
        for cls in CLASS_NAMES:
            assert f"f1_{cls}" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
