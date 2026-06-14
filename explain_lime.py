"""
explain_lime.py — Generate LIME explanations for multiple images.

Usage:
    python explain_lime.py --model efficientnet --checkpoint results/checkpoints/best_efficientnet_b3.pth
    python explain_lime.py --model mobilenet    --checkpoint results/checkpoints/best_mobilenet_v3.pth --n-images 5
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from src.data.dataset import load_and_normalize_csv
from src.models.model_zoo import build_model
from src.explainability.gradcam_lime import LIMEExplainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, required=True,
                        choices=["efficientnet", "resnet", "densenet", "mobilenet"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-csv",   type=str, default="data/raw/GroundTruth.csv")
    parser.add_argument("--images-dir", type=str, default="data/raw/images")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--n-images",   type=int, default=5,
                        help="Number of images to explain")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="LIME perturbation samples (higher = more accurate but slower)")
    parser.add_argument("--save-dir",   type=str, default="results/reports/lime")
    return parser.parse_args()


def main():
    args   = parse_args()
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Model  : {args.model}")
    print(f"Images : {args.n_images} (samples per image: {args.num_samples})\n")

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        args.model,
        ckpt.get("config", {"num_classes": 7, "dropout": 0.3, "pretrained": False})
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    run_name = Path(args.checkpoint).stem

    # ── Load image list ───────────────────────────────────────────────────────
    df       = load_and_normalize_csv(args.data_csv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    explainer = LIMEExplainer(model, device, image_size=args.image_size)

    # Pick one image per class for variety
    classes_seen = set()
    selected     = []
    for _, row in df.iterrows():
        if row["dx"] not in classes_seen:
            selected.append(row)
            classes_seen.add(row["dx"])
        if len(selected) >= args.n_images:
            break

    # ── Run LIME ──────────────────────────────────────────────────────────────
    for i, row in enumerate(selected):
        img_id  = row["image_id"]
        true_cls = row["dx"]
        img_path = Path(args.images_dir) / f"{img_id}.jpg"

        if not img_path.exists():
            print(f"  Skipping {img_id} — file not found")
            continue

        print(f"[{i+1}/{len(selected)}] Explaining {img_id} (true: {true_cls})...")
        img = Image.open(img_path).convert("RGB")

        explanation, pred_idx = explainer.explain(img, num_samples=args.num_samples)

        save_path = save_dir / f"lime_{run_name}_{true_cls}_{img_id}.png"
        explainer.plot(img, explanation, pred_idx, save_path, run_name)

    print(f"\nAll LIME explanations saved to: {save_dir}/")
    print("Open the PNG files to see which regions drove each prediction.")


if __name__ == "__main__":
    main()
