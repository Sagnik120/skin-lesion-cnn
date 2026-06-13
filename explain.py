"""
explain.py — Generate Grad-CAM and LIME explanations for the best model.

Usage:
    python explain.py --model efficientnet --checkpoint results/checkpoints/best_efficientnet_b3.pth
"""

import argparse
import torch
from pathlib import Path

from src.data.dataset import build_loaders
from src.models.model_zoo import build_model
from src.explainability.gradcam_lime import visualize_gradcam_batch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-csv",   type=str, default="data/raw/HAM10000_metadata.csv")
    parser.add_argument("--images-dir", type=str, default="data/raw/images")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--n-images",   type=int, default=8,  help="Number of Grad-CAM examples")
    return parser.parse_args()


def main():
    args   = parse_args()
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = build_model(args.model, ckpt.get("config", {"num_classes": 7, "dropout": 0.3, "pretrained": False}))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    _, val_loader, _ = build_loaders(
        args.data_csv, args.images_dir,
        image_size=args.image_size, batch_size=1,
    )

    run_name = Path(args.checkpoint).stem

    print(f"Generating Grad-CAM for {args.n_images} images...")
    visualize_gradcam_batch(
        model,
        val_loader,
        device=device,
        save_dir="results/reports",
        n_images=args.n_images,
        model_name=run_name,
    )

    print("\nExplainability outputs saved to results/reports/")
    print("Open the .png files to inspect which regions the model focuses on.")


if __name__ == "__main__":
    main()
