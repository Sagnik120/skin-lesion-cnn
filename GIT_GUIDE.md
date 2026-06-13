# 🔄 Git Commit Guide — skin-lesion-cnn

Your username: **Sagnik120**
Repo URL: `https://github.com/Sagnik120/skin-lesion-cnn`

---

## 🚀 First-time setup

```bash
# 1. Create the repo on GitHub.com (name it: skin-lesion-cnn)
#    Then come back here and run:

cd skin-lesion-cnn

git init
git remote add origin https://github.com/Sagnik120/skin-lesion-cnn.git
git branch -M main
```

---

## 📦 Commit 1 — Project initialization

```bash
git add README.md requirements.txt .gitignore CHANGELOG.md
git commit -m "feat: initialize project structure and docs"
git push -u origin main
```

---

## 📦 Commit 2 — Configs

```bash
git add configs/
git commit -m "feat: add training configs for all 4 CNN architectures"
git push
```

---

## 📦 Commit 3 — Data pipeline

```bash
git add src/data/
git commit -m "feat: add HAM10000 dataset class, augmentation, and dataloaders"
git push
```

---

## 📦 Commit 4 — Model zoo

```bash
git add src/models/
git commit -m "feat: add model zoo — EfficientNet, ResNet, DenseNet, MobileNet"
git push
```

---

## 📦 Commit 5 — Training infrastructure

```bash
git add src/training/
git commit -m "feat: add Focal Loss, Trainer with early stopping, AdamW + cosine scheduler"
git push
```

---

## 📦 Commit 6 — Evaluation pipeline

```bash
git add src/evaluation/
git commit -m "feat: add evaluator with accuracy, F1, ROC-AUC, confusion matrix, comparison"
git push
```

---

## 📦 Commit 7 — Explainability

```bash
git add src/explainability/
git commit -m "feat: add Grad-CAM and LIME explainability modules"
git push
```

---

## 📦 Commit 8 — Entry point scripts

```bash
git add train.py evaluate.py compare_models.py explain.py
git commit -m "feat: add CLI entry points for train, evaluate, compare, explain"
git push
```

---

## 📦 Commit 9 — Tests

```bash
git add tests/
git commit -m "test: add unit tests for dataset, models, loss functions, metrics"
git push
```

---

## 📦 Commit 10 — After training EfficientNet

```bash
# After running: python train.py --model efficientnet --config configs/efficientnet.yaml

git add results/plots/training_curves_efficientnet_b3.png
git add results/plots/cm_efficientnet_b3.png
git add results/plots/roc_efficientnet_b3.png
git commit -m "results: EfficientNet-B3 training complete — val_f1=X.XX"
git push
```

---

## 📦 Commit 11 — After training ResNet

```bash
git add results/plots/training_curves_resnet50.png
git add results/plots/cm_resnet50.png
git add results/plots/roc_resnet50.png
git commit -m "results: ResNet-50 training complete"
git push
```

---

## 📦 Commit 12 — After training DenseNet

```bash
git add results/plots/training_curves_densenet121.png
git add results/plots/cm_densenet121.png
git add results/plots/roc_densenet121.png
git commit -m "results: DenseNet-121 training complete"
git push
```

---

## 📦 Commit 13 — After training MobileNet

```bash
git add results/plots/training_curves_mobilenet_v3.png
git add results/plots/cm_mobilenet_v3.png
git add results/plots/roc_mobilenet_v3.png
git commit -m "results: MobileNet-V3 training complete"
git push
```

---

## 📦 Commit 14 — Model comparison

```bash
git add results/reports/model_comparison.csv
git add results/plots/model_comparison_bars.png
git add results/plots/per_class_f1_heatmap.png
git commit -m "results: model comparison complete — best model selected"
git push
```

---

## 📦 Commit 15 — Grad-CAM results

```bash
git add results/reports/gradcam_*.png
git commit -m "results: Grad-CAM explainability visualizations added"
git push
```

---

## 📦 Commit 16 — Final update

```bash
# Update CHANGELOG.md with real results
git add CHANGELOG.md README.md
git commit -m "docs: update changelog and README with final results"
git push
```

---

## 💡 Useful git commands

```bash
# Check what's staged
git status

# See all commits
git log --oneline

# See what changed
git diff

# Add everything (be careful — check .gitignore first!)
git add .

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Create a feature branch (optional)
git checkout -b feature/eda-notebook
git push -u origin feature/eda-notebook
```

---

## 📌 Tips for more commits

- After each experiment, commit the results plots
- Add a `notebooks/01_eda.ipynb` for exploratory analysis → another commit
- Add a `notebooks/02_results.ipynb` for result visualization → another commit
- Fix a bug → commit it as `fix: ...`
- Update README with actual results → commit as `docs: ...`
