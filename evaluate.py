"""
evaluate.py — Evaluate a single trained model on the test set.

Usage:
    python evaluate.py --model efficientnet --checkpoint results/checkpoints/best_efficientnet_b3.pth
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.data.dataset import build_loaders
from src.models.model_zoo import build_model
from src.evaluation.evaluator import (
    run_inference, compute_metrics, print_report,
    plot_confusion_matrix, plot_roc_curves,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-csv",   type=str, default="data/raw/HAM10000_metadata.csv")
    parser.add_argument("--images-dir", type=str, default="data/raw/images")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args   = parse_args()
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = build_model(args.model, ckpt.get("config", {"num_classes": 7, "dropout": 0.3, "pretrained": False}))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    _, _, test_loader = build_loaders(
        args.data_csv, args.images_dir,
        image_size=args.image_size, batch_size=args.batch_size
    )

    labels, preds, probs = run_inference(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs)
    print_report(labels, preds, metrics)

    run_name = Path(args.checkpoint).stem
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(labels, preds, f"results/plots/cm_{run_name}.png", run_name)
    plot_roc_curves(labels, probs, f"results/plots/roc_{run_name}.png", run_name)


if __name__ == "__main__":
    main()
