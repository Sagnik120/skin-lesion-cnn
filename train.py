"""
train.py — Main training entry point.

Usage:
    python train.py --model efficientnet --config configs/efficientnet.yaml
    python train.py --model resnet       --config configs/resnet.yaml
    python train.py --model densenet     --config configs/densenet.yaml
    python train.py --model mobilenet    --config configs/mobilenet.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path

from src.data.dataset import build_loaders, CLASS_NAMES
from src.models.model_zoo import build_model
from src.training.losses import build_loss, build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.evaluation.evaluator import (
    run_inference, compute_metrics, print_report,
    plot_confusion_matrix, plot_training_curves, plot_roc_curves
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on HAM10000")
    parser.add_argument("--model",  type=str, required=True,
                        choices=["efficientnet", "resnet", "densenet", "mobilenet"],
                        help="Which model to train")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    parser.add_argument("--data-csv", type=str,
                        default="data/raw/HAM10000_metadata.csv")
    parser.add_argument("--images-dir", type=str,
                        default="data/raw/images")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_cfg    = config["model"]
    data_cfg     = config["data"]
    train_cfg    = config["training"]
    log_cfg      = config["logging"]
    flat_config  = {**model_cfg, **train_cfg, **log_cfg}

    torch.manual_seed(args.seed)
    device = "mps" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model : {args.model}")
    print(f"Config: {args.config}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Building DataLoaders...")
    train_loader, val_loader, test_loader = build_loaders(
        metadata_csv  = args.data_csv,
        images_dir    = args.images_dir,
        image_size    = data_cfg["image_size"],
        batch_size    = data_cfg["batch_size"],
        num_workers   = data_cfg["num_workers"],
        seed          = args.seed,
    )
    print(f"  Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = build_model(args.model, model_cfg)

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    class_weights = train_loader.dataset.get_class_weights().to(device)
    loss_fn   = build_loss(train_cfg, class_weights)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)

    print(f"\nLoss     : {train_cfg['loss']}")
    print(f"Optimizer: AdamW lr={train_cfg['learning_rate']}")
    print(f"Scheduler: cosine_warmup (warmup={train_cfg['warmup_epochs']} epochs)")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"TRAINING {args.model.upper()}")
    print("=" * 60)

    trainer = Trainer(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        loss_fn         = loss_fn,
        optimizer       = optimizer,
        scheduler       = scheduler,
        config          = flat_config,
        device          = device,
        checkpoint_dir  = log_cfg["checkpoint_dir"],
    )
    history = trainer.fit()

    # ── Training curves ───────────────────────────────────────────────────────
    Path("results/plots").mkdir(parents=True, exist_ok=True)
    run_name = log_cfg.get("run_name", args.model)
    plot_training_curves(
        history,
        save_path  = f"results/plots/training_curves_{run_name}.png",
        model_name = run_name,
    )

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)

    # Load best checkpoint
    best_ckpt = Path(log_cfg["checkpoint_dir"]) / f"best_{run_name}.pth"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']} (val_f1={ckpt['val_f1']:.4f})")

    model.to(device)
    labels, preds, probs = run_inference(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs)
    print_report(labels, preds, metrics)

    plot_confusion_matrix(labels, preds, f"results/plots/cm_{run_name}.png", run_name)
    plot_roc_curves(labels, probs, f"results/plots/roc_{run_name}.png", run_name)

    print(f"\nDone! Run 'python compare_models.py' after training all 4 models.")


if __name__ == "__main__":
    main()
