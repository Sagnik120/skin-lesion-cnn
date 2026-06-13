"""
Model comparison — load all 4 trained checkpoints, evaluate on the test set,
and produce a summary table + radar chart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.data.dataset import CLASS_NAMES
from src.evaluation.evaluator import run_inference, compute_metrics


CHECKPOINT_DIR = Path("results/checkpoints")
PLOTS_DIR      = Path("results/plots")
REPORTS_DIR    = Path("results/reports")

MODEL_CONFIGS = {
    "EfficientNet-B3": {"checkpoint": "best_efficientnet_b3.pth", "model_key": "efficientnet"},
    "ResNet-50":       {"checkpoint": "best_resnet50.pth",        "model_key": "resnet"},
    "DenseNet-121":    {"checkpoint": "best_densenet121.pth",     "model_key": "densenet"},
    "MobileNet-V3":    {"checkpoint": "best_mobilenet_v3.pth",    "model_key": "mobilenet"},
}


def compare_models(
    test_loader,
    device: str = "cuda",
) -> pd.DataFrame:
    """
    Load all checkpoints, run inference, and return a comparison DataFrame.
    """
    from src.models.model_zoo import build_model

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for display_name, cfg in MODEL_CONFIGS.items():
        ckpt_path = CHECKPOINT_DIR / cfg["checkpoint"]
        if not ckpt_path.exists():
            print(f"Checkpoint not found: {ckpt_path} — skipping {display_name}")
            continue

        print(f"\nEvaluating {display_name}...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model = build_model(cfg["model_key"], checkpoint.get("config", {"num_classes": 7, "dropout": 0.3, "pretrained": False}))
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)

        labels, preds, probs = run_inference(model, test_loader, device)
        metrics = compute_metrics(labels, preds, probs)

        row = {"Model": display_name}
        row.update({
            "Accuracy":      metrics["accuracy"],
            "F1 (macro)":    metrics["f1_macro"],
            "F1 (weighted)": metrics["f1_weighted"],
            "ROC-AUC":       metrics["roc_auc_macro"],
        })
        # Per-class F1
        for cls in CLASS_NAMES:
            row[f"F1_{cls}"] = metrics.get(f"f1_{cls}", float("nan"))

        results.append(row)
        print(f"  Accuracy={metrics['accuracy']:.4f} | F1={metrics['f1_macro']:.4f} | AUC={metrics['roc_auc_macro']:.4f}")

    df = pd.DataFrame(results).set_index("Model")

    # Save CSV
    csv_path = REPORTS_DIR / "model_comparison.csv"
    df.to_csv(csv_path)
    print(f"\nComparison table saved → {csv_path}")

    # Print table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON — TEST SET")
    print("=" * 70)
    print(df[["Accuracy", "F1 (macro)", "F1 (weighted)", "ROC-AUC"]].to_string())

    best_model = df["F1 (macro)"].idxmax()
    print(f"\n🏆 Best model by Macro F1: {best_model} = {df.loc[best_model, 'F1 (macro)']:.4f}")

    # Plots
    _plot_comparison_bars(df, PLOTS_DIR / "model_comparison_bars.png")
    _plot_per_class_heatmap(df, PLOTS_DIR / "per_class_f1_heatmap.png")

    return df


def _plot_comparison_bars(df: pd.DataFrame, save_path: Path):
    """Bar chart comparing 4 models on 4 metrics."""
    metrics = ["Accuracy", "F1 (macro)", "F1 (weighted)", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]

    for i, (model_name, row) in enumerate(df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=model_name, color=colors[i % len(colors)])

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Test Set")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved → {save_path}")


def _plot_per_class_heatmap(df: pd.DataFrame, save_path: Path):
    """Heatmap of per-class F1 scores across all models."""
    import seaborn as sns

    f1_cols = [f"F1_{cls}" for cls in CLASS_NAMES]
    sub = df[f1_cols].rename(columns={f"F1_{c}": c for c in CLASS_NAMES})

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(sub, annot=True, fmt=".2f", cmap="YlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Per-class F1 Score by Model")
    ax.set_xlabel("Skin Lesion Class")
    ax.set_ylabel("Model")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-class heatmap saved → {save_path}")
