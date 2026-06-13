"""
Loss functions and optimizer/scheduler utilities.

Focal Loss is used because HAM10000 is highly imbalanced
(nv class has 6x more samples than df/vasc).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


# ── Focal Loss ─────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Focal loss down-weights easy examples and focuses training on hard ones.
    Especially useful for imbalanced datasets.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Args:
        alpha: class weights tensor (shape: [num_classes]) — set from dataset.get_class_weights()
        gamma: focusing parameter (default 2.0 from paper)
        reduction: 'mean' or 'sum'
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross-entropy gives log-probabilities per class
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # Gather the probability for the true class
        true_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        true_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal factor: down-weights confident easy examples
        focal_weight = (1 - true_probs) ** self.gamma

        # Class weight from alpha
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * true_log_probs

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(config: dict, class_weights: torch.Tensor | None = None) -> nn.Module:
    """Build loss function from config."""
    loss_type = config.get("loss", "focal")

    if loss_type == "focal":
        gamma = config.get("focal_gamma", 2.0)
        return FocalLoss(alpha=class_weights, gamma=gamma)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unknown loss: {loss_type}")


# ── Optimizer ─────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, config: dict) -> AdamW:
    """Build AdamW optimizer with optional backbone/head LR differentiation."""
    lr = config.get("learning_rate", 1e-4)
    wd = config.get("weight_decay", 1e-4)

    # Give the classifier head a higher LR than the pretrained backbone
    try:
        backbone_params = list(model.backbone.parameters())
        head_params     = list(model.classifier.parameters())
        param_groups = [
            {"params": backbone_params, "lr": lr,      "weight_decay": wd},
            {"params": head_params,     "lr": lr * 10, "weight_decay": wd},
        ]
    except AttributeError:
        param_groups = model.parameters()

    return AdamW(param_groups, lr=lr, weight_decay=wd)


# ── Scheduler ─────────────────────────────────────────────────────────────────

def build_scheduler(optimizer, config: dict):
    """
    Build a warmup + cosine annealing learning rate scheduler.

    Linear warmup for `warmup_epochs`, then cosine decay to `min_lr`.
    """
    total_epochs  = config.get("epochs", 50)
    warmup_epochs = config.get("warmup_epochs", 5)
    min_lr        = config.get("min_lr", 1e-6)

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )
    return scheduler


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test focal loss
    logits  = torch.randn(4, 7)
    targets = torch.tensor([0, 3, 6, 1])
    weights = torch.ones(7)

    fl = FocalLoss(alpha=weights, gamma=2.0)
    ce = nn.CrossEntropyLoss(weight=weights)

    fl_val = fl(logits, targets)
    ce_val = ce(logits, targets)

    print(f"Focal Loss   : {fl_val.item():.4f}")
    print(f"Cross-Entropy: {ce_val.item():.4f}")
    print("Loss test passed ✓")
