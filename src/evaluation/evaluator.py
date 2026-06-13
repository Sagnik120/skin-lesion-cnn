"""
Evaluation module — compute and visualize all metrics for a trained model.

Metrics computed:
- Accuracy (overall + per-class)
- Macro / Weighted F1-score
- ROC-AUC (per class + macro OvR)
- Confusion matrix (normalized heatmap)
- Full classification report
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from src.data.dataset import CLASS_NAMES


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run full inference on a DataLoader.

    Returns:
        labels    : true class indices, shape (N,)
        preds     : predicted class indices, shape (N,)
        probs     : softmax probabilities, shape (N, num_classes)
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels, _ in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.append(probs)

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    labels: np.ndarray,
    preds:  np.ndarray,
    probs:  np.ndarray,
) -> dict:
    """Return a flat dict of all evaluation metrics."""
    acc_overall  = accuracy_score(labels, preds)
    f1_macro     = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_weighted  = f1_score(labels, preds, average="weighted", zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None,       zero_division=0)

    # ROC-AUC (one-vs-rest, handles multi-class)
    try:
        roc_auc_macro   = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        roc_auc_per_cls = roc_auc_score(
            labels, probs, multi_class="ovr", average=None,
            labels=list(range(len(CLASS_NAMES)))
        )
    except ValueError:
        roc_auc_macro   = float("nan")
        roc_auc_per_cls = [float("nan")] * len(CLASS_NAMES)

    # Per-class accuracy
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    metrics = {
        "accuracy":      acc_overall,
        "f1_macro":      f1_macro,
        "f1_weighted":   f1_weighted,
        "roc_auc_macro": roc_auc_macro,
    }
    for i, cls in enumerate(CLASS_NAMES):
        metrics[f"f1_{cls}"]       = f1_per_class[i]
        metrics[f"acc_{cls}"]      = per_class_acc[i]
        metrics[f"roc_auc_{cls}"]  = roc_auc_per_cls[i] if not isinstance(roc_auc_per_cls, float) else float("nan")

    return metrics


def print_report(labels: np.ndarray, preds: np.ndarray, metrics: dict):
    """Pretty-print the evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Overall Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1-score   : {metrics['f1_macro']:.4f}")
    print(f"Weighted F1-score: {metrics['f1_weighted']:.4f}")
    print(f"Macro ROC-AUC    : {metrics['roc_auc_macro']:.4f}")
    print("\nPer-class metrics:")
    print(f"{'Class':<10} {'F1':>8} {'Accuracy':>10} {'ROC-AUC':>10}")
    print("-" * 42)
    for cls in CLASS_NAMES:
        f1      = metrics.get(f"f1_{cls}", float("nan"))
        acc_c   = metrics.get(f"acc_{cls}", float("nan"))
        roc_c   = metrics.get(f"roc_auc_{cls}", float("nan"))
        print(f"{cls:<10} {f1:>8.4f} {acc_c:>10.4f} {roc_c:>10.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels:    np.ndarray,
    preds:     np.ndarray,
    save_path: str | Path,
    model_name: str = "",
):
    """Plot and save a normalized confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds, normalize="true")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def plot_training_curves(
    history:   dict,
    save_path: str | Path,
    model_name: str = "",
):
    """Plot loss, accuracy, and F1 curves from training history."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")

    axes[2].plot(epochs, history["train_f1"], label="Train")
    axes[2].plot(epochs, history["val_f1"],   label="Val")
    axes[2].set_title("Macro F1"); axes[2].legend(); axes[2].set_xlabel("Epoch")

    fig.suptitle(f"Training Curves — {model_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved → {save_path}")


def plot_roc_curves(
    labels:    np.ndarray,
    probs:     np.ndarray,
    save_path: str | Path,
    model_name: str = "",
):
    """Plot per-class ROC curves."""
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    labels_bin = label_binarize(labels, classes=list(range(len(CLASS_NAMES))))

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CLASS_NAMES)))

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        try:
            auc = roc_auc_score(labels_bin[:, i], probs[:, i])
        except ValueError:
            auc = float("nan")
        ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{cls} (AUC={auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {model_name}")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curves saved → {save_path}")
