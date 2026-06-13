"""
SkinLesionDataset — PyTorch Dataset for HAM10000.

Handles:
- Loading images from disk by image_id
- Applying train/val/test augmentations via albumentations
- Returning (image_tensor, label_int, image_id) triples
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


# ── Class definitions ────────────────────────────────────────────────────────

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "vasc":  "Vascular lesion",
}

# ImageNet normalisation stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_train_transform(image_size: int = 224):
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.7),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.GaussNoise(var_limit=(10, 50), p=0.5),
        ], p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transform(image_size: int = 224):
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class SkinLesionDataset(Dataset):
    """PyTorch Dataset for HAM10000 dermoscopy images."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str | Path,
        transform=None,
    ):
        """
        Args:
            df: DataFrame with columns [image_id, dx]
            images_dir: Path to folder containing .jpg images
            transform: albumentations transform pipeline
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        label = CLASS_TO_IDX[row["dx"]]

        img_path = self.images_dir / f"{image_id}.jpg"
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(label, dtype=torch.long), image_id

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for weighted loss."""
        counts = self.df["dx"].map(CLASS_TO_IDX).value_counts().sort_index()
        weights = 1.0 / counts.values
        weights = weights / weights.sum() * len(CLASS_NAMES)
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> list[float]:
        """Per-sample weights for WeightedRandomSampler (balances mini-batches)."""
        class_counts = self.df["dx"].value_counts().to_dict()
        return [1.0 / class_counts[row["dx"]] for _, row in self.df.iterrows()]


# ── Data splitting ─────────────────────────────────────────────────────────────

def make_splits(
    metadata_csv: str | Path,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test split.

    Returns:
        (train_df, val_df, test_df) — each with [image_id, dx] columns.
    """
    df = pd.read_csv(metadata_csv)[["image_id", "dx"]]

    # Remove duplicate image_ids (same lesion photographed multiple times)
    df = df.drop_duplicates(subset="image_id")

    train_val, test = train_test_split(
        df, test_size=test_ratio, stratify=df["dx"], random_state=seed
    )
    val_rel = val_ratio / (1 - test_ratio)
    train, val = train_test_split(
        train_val, test_size=val_rel, stratify=train_val["dx"], random_state=seed
    )

    print(f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ── DataLoaders ───────────────────────────────────────────────────────────────

def build_loaders(
    metadata_csv: str | Path,
    images_dir: str | Path,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders for HAM10000.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_df, val_df, test_df = make_splits(metadata_csv, seed=seed)

    train_ds = SkinLesionDataset(train_df, images_dir, get_train_transform(image_size))
    val_ds   = SkinLesionDataset(val_df,   images_dir, get_val_transform(image_size))
    test_ds  = SkinLesionDataset(test_df,  images_dir, get_val_transform(image_size))

    # Weighted sampler ensures each mini-batch has balanced classes
    train_sampler = None
    if use_weighted_sampler:
        sample_weights = train_ds.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    csv_path = "data/raw/HAM10000_metadata.csv"
    img_dir  = "data/raw/images"

    if not Path(csv_path).exists():
        print(f"ERROR: {csv_path} not found. Run: python src/data/download_data.py")
        sys.exit(1)

    train_loader, val_loader, test_loader = build_loaders(csv_path, img_dir)

    images, labels, ids = next(iter(train_loader))
    print(f"Batch shape : {images.shape}")
    print(f"Labels      : {labels[:8].tolist()}")
    print(f"Image IDs   : {ids[:4]}")
    print("DataLoader check passed ✓")
