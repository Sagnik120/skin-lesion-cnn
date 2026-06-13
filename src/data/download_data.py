"""
Download HAM10000 dataset from Kaggle.

Prerequisites:
  1. Install kaggle: pip install kaggle
  2. Get your API key from https://www.kaggle.com/settings -> API -> Create New Token
  3. Place kaggle.json in ~/.kaggle/kaggle.json  (Linux/Mac)
                       or %USERPROFILE%\.kaggle\kaggle.json (Windows)
  4. Run: python src/data/download_data.py
"""

import os
import zipfile
import shutil
from pathlib import Path


DATA_DIR = Path("data/raw")
KAGGLE_DATASET = "kmader/skin-lesion-analysis-toward-melanoma-detection"


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading HAM10000 from Kaggle...")
    print("Dataset:", KAGGLE_DATASET)
    print("Saving to:", DATA_DIR.resolve())

    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            KAGGLE_DATASET,
            path=str(DATA_DIR),
            unzip=True,
            quiet=False,
        )
        print("\n✓ Download complete!")
        _verify_and_organize()
    except ImportError:
        print("ERROR: kaggle package not installed. Run: pip install kaggle")
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        print("\nManual download option:")
        print("  1. Go to https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection")
        print("  2. Download and extract to data/raw/")
        print("  3. Make sure you have: data/raw/HAM10000_images_part_1/, data/raw/HAM10000_images_part_2/")
        print("  4. And: data/raw/HAM10000_metadata.csv")
        raise


def _verify_and_organize():
    """Check key files exist and flatten image folders."""
    images_dir = DATA_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    # HAM10000 comes in two parts — merge them
    for part in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        part_dir = DATA_DIR / part
        if part_dir.exists():
            print(f"Merging {part} -> data/raw/images/")
            for img in part_dir.glob("*.jpg"):
                shutil.move(str(img), str(images_dir / img.name))
            part_dir.rmdir()

    # Check metadata
    meta = DATA_DIR / "HAM10000_metadata.csv"
    if not meta.exists():
        print("WARNING: HAM10000_metadata.csv not found. Check your download.")
    else:
        import pandas as pd
        df = pd.read_csv(meta)
        print(f"\nDataset summary:")
        print(f"  Total images: {len(df)}")
        print(f"  Classes: {df['dx'].nunique()}")
        print(f"  Class distribution:")
        print(df['dx'].value_counts().to_string(index=True))

    print(f"\nImages saved to: {images_dir.resolve()}")


if __name__ == "__main__":
    download()
