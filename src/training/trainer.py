"""
Trainer — full training loop with:
- Epoch-level train/val loop
- Early stopping on val_f1_macro
- Backbone freeze/unfreeze schedule
- MLflow experiment tracking
- Best model checkpoint saving
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not installed — logging disabled. Run: pip install mlflow")


class EarlyStopping:
    """Stops training when a metric stops improving for `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_score = None
        self.counter    = 0
        self.triggered  = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score > self.best_score + self.min_delta) if self.mode == "max" \
                   else (score < self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False


class Trainer:
    """
    Train a SkinLesionModel with configurable loss, optimizer, and scheduler.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, config, device)
        history = trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        loss_fn:      nn.Module,
        optimizer,
        scheduler,
        config:       dict,
        device:       str = "mps",
        checkpoint_dir: str = "results/checkpoints",
    ):
        self.model          = model.to(device)
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.loss_fn        = loss_fn.to(device)
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.config         = config
        self.device         = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", 10),
            mode="max",
        )

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "train_f1":   [], "val_f1":   [],
            "lr": [],
        }

    def fit(self) -> dict:
        epochs                = self.config.get("epochs", 50)
        freeze_epochs         = self.config.get("freeze_backbone_epochs", 3)
        model_name            = getattr(self.model, "model_name", "model")
        run_name              = self.config.get("run_name", model_name)
        experiment_name       = self.config.get("experiment_name", "skin_lesion_cnn")

        best_f1    = 0.0
        best_ckpt  = self.checkpoint_dir / f"best_{run_name.replace(' ', '_')}.pth"

        # ── MLflow setup ──────────────────────────────────────────────────────
        use_mlflow = MLFLOW_AVAILABLE
        if use_mlflow:
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "model": model_name,
                "epochs": epochs,
                "lr": self.config.get("learning_rate", 1e-4),
                "batch_size": self.train_loader.batch_size,
                "loss": self.config.get("loss", "focal"),
                "freeze_epochs": freeze_epochs,
            })

        # ── Training loop ─────────────────────────────────────────────────────
        try:
            for epoch in range(1, epochs + 1):
                # Backbone scheduling
                if epoch == 1 and freeze_epochs > 0:
                    self.model.freeze_backbone()
                elif epoch == freeze_epochs + 1:
                    self.model.unfreeze_backbone()

                t0 = time.time()

                train_loss, train_acc, train_f1 = self._run_epoch(train=True)
                val_loss,   val_acc,   val_f1   = self._run_epoch(train=False)

                current_lr = self.optimizer.param_groups[0]["lr"]
                self.scheduler.step()

                elapsed = time.time() - t0

                # Record history
                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_acc"].append(val_acc)
                self.history["train_f1"].append(train_f1)
                self.history["val_f1"].append(val_f1)
                self.history["lr"].append(current_lr)

                # MLflow logging
                if use_mlflow:
                    mlflow.log_metrics({
                        "train_loss": train_loss, "val_loss": val_loss,
                        "train_acc":  train_acc,  "val_acc":  val_acc,
                        "train_f1":   train_f1,   "val_f1":   val_f1,
                        "lr":         current_lr,
                    }, step=epoch)

                print(
                    f"Epoch {epoch:03d}/{epochs} | "
                    f"Loss {train_loss:.4f}/{val_loss:.4f} | "
                    f"Acc {train_acc:.3f}/{val_acc:.3f} | "
                    f"F1 {train_f1:.3f}/{val_f1:.3f} | "
                    f"LR {current_lr:.2e} | {elapsed:.1f}s"
                )

                # Save best model
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    torch.save({
                        "epoch":       epoch,
                        "model_state": self.model.state_dict(),
                        "val_f1":      val_f1,
                        "val_acc":     val_acc,
                        "config":      self.config,
                    }, best_ckpt)
                    print(f"  ✓ New best model saved → val_f1 = {val_f1:.4f}")

                # Early stopping
                if self.early_stopping(val_f1):
                    print(f"Early stopping triggered at epoch {epoch} (best val_f1={best_f1:.4f})")
                    break

        finally:
            if use_mlflow:
                mlflow.log_metric("best_val_f1", best_f1)
                mlflow.log_artifact(str(best_ckpt))
                mlflow.end_run()

        print(f"\nTraining complete. Best val F1 = {best_f1:.4f}")
        print(f"Checkpoint saved at: {best_ckpt}")
        return self.history

    def _run_epoch(self, train: bool) -> tuple[float, float, float]:
        self.model.train(train)
        loader = self.train_loader if train else self.val_loader
        desc   = "Train" if train else "Val  "

        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.set_grad_enabled(train):
            for images, labels, _ in tqdm(loader, desc=desc, leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss   = self.loss_fn(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * len(images)
                preds = logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().tolist())

        n    = len(loader.dataset)
        loss = total_loss / n
        acc  = accuracy_score(all_labels, all_preds)
        f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return loss, acc, f1
