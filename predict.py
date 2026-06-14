"""
predict.py — Predict skin lesion class for a single image.

Usage:
    python predict.py --image path/to/image.jpg
    python predict.py --image path/to/image.jpg --model mobilenet --checkpoint results/checkpoints/best_mobilenet_v3.pth
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from src.models.model_zoo import build_model
from src.data.dataset import (
    get_val_transform, CLASS_NAMES, IDX_TO_CLASS, CLASS_DESCRIPTIONS,
    IMAGENET_MEAN, IMAGENET_STD
)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict skin lesion from a single image")
    parser.add_argument("--image",      type=str, required=True,
                        help="Path to input image (.jpg or .png)")
    parser.add_argument("--model",      type=str, default="efficientnet",
                        choices=["efficientnet", "resnet", "densenet", "mobilenet"])
    parser.add_argument("--checkpoint", type=str,
                        default="results/checkpoints/best_efficientnet_b3.pth")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--save-plot",  type=str, default=None,
                        help="Optional: save prediction plot to this path")
    return parser.parse_args()


def predict(image_path, model, device, image_size=224):
    """Run inference on a single image. Returns (predicted_class, confidence, all_probs)."""
    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img)

    transform = get_val_transform(image_size)
    tensor = transform(image=img_array)["image"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

    pred_idx   = probs.argmax()
    pred_class = IDX_TO_CLASS[pred_idx]
    confidence = probs[pred_idx]

    return pred_class, confidence, probs


def print_results(pred_class, confidence, probs, image_path):
    """Pretty print prediction results."""
    print("\n" + "=" * 50)
    print("SKIN LESION PREDICTION")
    print("=" * 50)
    print(f"Image     : {image_path}")
    print(f"Prediction: {pred_class.upper()} — {CLASS_DESCRIPTIONS[pred_class]}")
    print(f"Confidence: {confidence * 100:.1f}%")
    print("\nAll class probabilities:")
    print("-" * 40)

    # Sort by probability descending
    sorted_idx = probs.argsort()[::-1]
    for idx in sorted_idx:
        cls  = CLASS_NAMES[idx]
        prob = probs[idx]
        bar  = "█" * int(prob * 30)
        print(f"  {cls:<8} {prob*100:5.1f}%  {bar}")
    print("=" * 50)


def save_prediction_plot(image_path, pred_class, confidence, probs, save_path):
    """Save a visual prediction report."""
    img = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: original image
    axes[0].imshow(img)
    axes[0].set_title(
        f"Prediction: {pred_class.upper()}\n"
        f"{CLASS_DESCRIPTIONS[pred_class]}\n"
        f"Confidence: {confidence*100:.1f}%",
        fontsize=12, fontweight="bold",
        color="darkred" if pred_class == "mel" else "darkgreen"
    )
    axes[0].axis("off")

    # Right: probability bar chart
    sorted_idx = probs.argsort()[::-1]
    sorted_cls  = [CLASS_NAMES[i] for i in sorted_idx]
    sorted_prob = [probs[i] for i in sorted_idx]
    colors = ["#e74c3c" if c == pred_class else "#3498db" for c in sorted_cls]

    axes[1].barh(sorted_cls[::-1], sorted_prob[::-1], color=colors[::-1])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Class Probabilities")
    for i, (cls, prob) in enumerate(zip(sorted_cls[::-1], sorted_prob[::-1])):
        axes[1].text(prob + 0.01, i, f"{prob*100:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPrediction plot saved → {save_path}")


def main():
    args = parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        return

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model} from {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = build_model(
        args.model,
        ckpt.get("config", {"num_classes": 7, "dropout": 0.3, "pretrained": False})
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    pred_class, confidence, probs = predict(
        args.image, model, device, args.image_size
    )

    print_results(pred_class, confidence, probs, args.image)

    if args.save_plot:
        Path(args.save_plot).parent.mkdir(parents=True, exist_ok=True)
        save_prediction_plot(args.image, pred_class, confidence, probs, args.save_plot)


if __name__ == "__main__":
    main()
