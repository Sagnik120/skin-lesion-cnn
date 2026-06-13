# Changelog

All notable changes to this project are documented here.
Each section = one git commit you should make.

## [Unreleased]

## [0.1.0] - Project Initialization
- Created project structure
- Added README, requirements, gitignore
- Set up configs directory

## [0.2.0] - Data Pipeline
- Added HAM10000 dataset downloader
- Implemented SkinLesionDataset class
- Added augmentation pipeline with albumentations
- Added stratified train/val/test split

## [0.3.0] - Model Zoo
- Implemented EfficientNet-B3 wrapper
- Implemented ResNet-50 wrapper
- Implemented DenseNet-121 wrapper
- Implemented MobileNet-V3 wrapper

## [0.4.0] - Training Infrastructure
- Added Focal Loss for class imbalance
- Added Trainer class with early stopping
- Added AdamW + cosine LR scheduler
- Added MLflow experiment tracking

## [0.5.0] - Evaluation Pipeline
- Added multi-metric evaluator
- Added confusion matrix visualization
- Added ROC-AUC per class
- Added model comparison table

## [0.6.0] - Explainability
- Added Grad-CAM implementation
- Added LIME explanations
- Added prediction report generator

## [0.7.0] - EfficientNet Training Run
- Trained EfficientNet-B3 for 30 epochs
- Results saved to results/plots/

## [0.8.0] - ResNet Training Run
- Trained ResNet-50 for 30 epochs
- Compared with EfficientNet

## [0.9.0] - DenseNet Training Run
- Trained DenseNet-121
- Updated comparison table

## [0.10.0] - MobileNet Training Run
- Trained MobileNet-V3
- Final model selection

## [1.0.0] - Final Evaluation
- Best model: (to be filled)
- Grad-CAM visualizations added
- Final report generated
