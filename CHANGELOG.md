# Changelog

All notable changes to this project are documented here.

---

## [1.0.0] - 2026-06-14 — Project Complete 🎉

### Final Results
| Model | Accuracy | Macro F1 | Weighted F1 | ROC-AUC |
|-------|----------|----------|-------------|---------|
| EfficientNet-B3 | 78.1% | 0.729 | 0.798 | 0.954 |
| MobileNet-V3 | 76.9% | **0.748** | 0.789 | 0.962 |
| DenseNet-121 | 71.5% | 0.725 | 0.743 | **0.968** |
| ResNet-50 | 65.4% | 0.691 | 0.688 | 0.952 |

🏆 **Best by Macro F1: MobileNet-V3 (0.748)**
🏆 **Best by Accuracy: EfficientNet-B3 (78.1%)**

---

## [0.10.0] - 2026-06-14 — Explainability Complete

### Added
- Grad-CAM heatmaps for EfficientNet-B3 and MobileNet-V3
- LIME superpixel explanations — model correctly focuses on lesion region
- `explain_lime.py` — standalone LIME script for any model
- `predict.py` — single image prediction with confidence scores and bar chart

---

## [0.9.0] - 2026-06-14 — All Models Trained

### EfficientNet-B3
- 50 epochs, MPS device
- Best val F1 = 0.800 (epoch 32)
- Test accuracy = 82.0%, ROC-AUC = 0.969

### ResNet-50
- Test accuracy = 65.4%, ROC-AUC = 0.952
- Struggled with mel class (F1=0.43)

### DenseNet-121
- Test accuracy = 71.5%, ROC-AUC = 0.968

### MobileNet-V3
- Test accuracy = 76.9%, ROC-AUC = 0.962
- Best Macro F1 overall (0.748)

---

## [0.8.0] - 2026-06-14 — EDA Notebook

### Added
- `notebooks/01_eda.ipynb` with class distribution, sample images, pixel stats
- Key finding: nv class dominates at 67%
- Imbalance ratio: 67x between nv and df classes

---

## [0.7.0] - 2026-06-13 — Training Infrastructure Fixed

### Fixed
- Switched device from CPU to MPS (Apple Silicon) — 5x speedup
- Fixed albumentations v2 API
- Fixed Grad-CAM inplace op issue with MobileNet
- Fixed Grad-CAM target layers for EfficientNet and MobileNet

---

## [0.6.0] - 2026-06-13 — Dataset Pipeline

### Added
- HAM10000 dataset via Kaggle API
- Fixed CSV loader for one-hot encoded GroundTruth.csv
- Stratified 70/15/15 split (7009/1503/1503)

---

## [0.5.0] - 2026-06-13 — Explainability Modules

### Added
- Grad-CAM with hook-based gradients
- LIME superpixel attribution
- Batch Grad-CAM visualization grid

---

## [0.4.0] - 2026-06-13 — Evaluation Pipeline

### Added
- Accuracy, F1, ROC-AUC, confusion matrix
- Multi-model comparison table and plots

---

## [0.3.0] - 2026-06-13 — Training Infrastructure

### Added
- Focal Loss from scratch (gamma=2.0)
- Trainer with early stopping and MLflow tracking
- AdamW + cosine warmup scheduler
- Backbone freeze/unfreeze schedule

---

## [0.2.0] - 2026-06-13 — Model Zoo

### Added
- EfficientNet-B3, ResNet-50, DenseNet-121, MobileNet-V3
- All pretrained on ImageNet via timm

---

## [0.1.0] - 2026-06-13 — Project Initialization

### Added
- Project structure, README, requirements, configs
- Unit tests for dataset, models, loss, metrics