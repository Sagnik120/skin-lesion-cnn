"""
compare_models.py — Load all 4 trained checkpoints and compare on the test set.

Usage:
    python compare_models.py
"""

import torch
from src.data.dataset import build_loaders
from src.evaluation.compare import compare_models


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    _, _, test_loader = build_loaders(
        "data/raw/GroundTruth.csv",
        "data/raw/images",
        image_size=224,
        batch_size=32,
    )

    df = compare_models(test_loader, device=device)

    print("\nFull comparison table:")
    print(df.to_string())


if __name__ == "__main__":
    main()
