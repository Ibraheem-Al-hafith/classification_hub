"""Utility module for training, evaluating, and plotting PyTorch models.

This module provides a generic, reusable framework for binary and multi-class
classification tasks following the DRY principle. It tracks accuracy, F1 score,
and loss across training/validation loops, exports performance curves, features
a standalone evaluation engine, and optimizes VRAM usage to prevent OOM errors.
"""

import gc
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LRScheduler,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainerConfig:
    """Configuration dataclass governing hyperparameters and training behavior."""

    # Computational settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization Hyperparameters
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-2

    # Learning Rate Scheduler Options: 'cosine', 'reduce_on_plateau', 'exponential', None
    scheduler_type: Optional[str] = "cosine"
    scheduler_monitor_metric: str = "val_loss"

    # Task definition configuration: 'binary' or 'multiclass'
    task_type: str = "multiclass"

    # Checkpointing & Saving paths
    checkpoint_dir: str = "./checkpoints"
    best_model_filename: str = "best_model.pt"
    last_model_filename: str = "last_model.pt"

    # Optional kwargs for schedulers passed dynamically
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)


class Trainer:
    """A generic PyTorch Trainer handling metrics tracking, checkpointing, and evaluation."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Initializes the Trainer with model, config, loss, metrics tracking, and data loaders."""
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        # Initialize Scheduler
        self.scheduler = self._get_scheduler()

        # Internal state tracking
        self.best_val_loss = float("inf")
        self.history: Dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "train_f1": [], "val_f1": []
        }
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def _get_scheduler(self) -> Optional[Union[LRScheduler, ReduceLROnPlateau]]:
        """Maps configuration strings to actual PyTorch LR scheduler instances."""
        if not self.config.scheduler_type:
            return None

        scheduler_map: Dict[str, Callable[..., Any]] = {
            "cosine": lambda opt: CosineAnnealingLR(
                opt, T_max=self.config.epochs, **self.config.scheduler_kwargs
            ),
            "reduce_on_plateau": lambda opt: ReduceLROnPlateau(
                opt, mode="min", **self.config.scheduler_kwargs
            ),
            "exponential": lambda opt: ExponentialLR(
                opt, gamma=0.95, **self.config.scheduler_kwargs
            ),
        }

        normalized_type = self.config.scheduler_type.lower().replace(" ", "_")
        if normalized_type not in scheduler_map:
            raise ValueError(f"Unknown scheduler type: '{self.config.scheduler_type}'")

        return scheduler_map[normalized_type](self.optimizer)

    @staticmethod
    def _clear_vram_cache() -> None:
        """Forces unreferencing garbage collection and flushes back-end accelerator caches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.mps.is_available():
            torch.mps.empty_cache()

    @staticmethod
    def _calculate_metrics(
        outputs: torch.Tensor, targets: torch.Tensor, task_type: str
    ) -> Tuple[float, float]:
        """Computes multi-class or binary accuracy and macro F1 scores strictly on device."""
        if task_type == "binary":
            preds = (torch.sigmoid(outputs) >= 0.5).long()
            targets = targets.long()
            num_classes = 2
        else:
            preds = torch.argmax(outputs, dim=1)
            targets = targets.long()
            num_classes = outputs.shape[1]

        # Calculate Global Accuracy
        correct = (preds == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total if total > 0 else 0.0

        # Calculate Macro F1 Score natively
        f1_classes = []
        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().item()
            fp = ((preds == c) & (targets != c)).sum().item()
            fn = ((preds != c) & (targets == c)).sum().item()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_classes.append(f1)

        macro_f1 = sum(f1_classes) / num_classes
        return accuracy, macro_f1

    def _run_epoch(self, is_train: bool = True, alternative_loader: Optional[DataLoader] = None) -> Tuple[float, float, float]:
        """Runs a single training or validation epoch over a given DataLoader."""
        self.model.train(is_train)
        
        # Resolve which loader instance to pull batches from
        if alternative_loader is not None:
            loader = alternative_loader
        else:
            loader = self.train_loader if is_train else self.val_loader
        
        if loader is None:
            return 0.0, 0.0, 0.0

        running_loss = 0.0
        all_outputs = []
        all_targets = []
        
        # Determine display string context descriptor
        if alternative_loader is not None:
            desc = "Evaluating"
        else:
            desc = "Training" if is_train else "Validation"

        with torch.set_grad_enabled(is_train):
            pbar = tqdm(loader, desc=desc, leave=False)
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                
                # Detach metrics to avoid retaining gradients in memory
                all_outputs.append(outputs.detach())
                all_targets.append(targets.detach())

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(loader.dataset)
        epoch_outputs = torch.cat(all_outputs, dim=0)
        epoch_targets = torch.cat(all_targets, dim=0)
        
        epoch_acc, epoch_f1 = self._calculate_metrics(
            epoch_outputs, epoch_targets, self.config.task_type
        )

        # Explicitly delete references and flush memory cache to prevent fragmentation
        del all_outputs, all_targets, epoch_outputs, epoch_targets
        self._clear_vram_cache()

        return epoch_loss, epoch_acc, epoch_f1

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, float]:
        """Evaluates the model on a provided dataset loader."""
        loss, accuracy, f1_score = self._run_epoch(is_train=False, alternative_loader=data_loader)
        return loss, accuracy, f1_score

    def fit(self) -> Dict[str, list[float]]:
        """Executes the training loop lifecycle while recording history metrics."""
        print(f"Starting pipeline on device: {self.device} (Task: {self.config.task_type})")
        
        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc, train_f1 = self._run_epoch(is_train=True)
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["train_f1"].append(train_f1)

            val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
            if self.val_loader:
                val_loss, val_acc, val_f1 = self.evaluate(self.val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_f1"].append(val_f1)

            # Informative console logging
            status = (
                f"Epoch [{epoch}/{self.config.epochs}] -> "
                f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}% | F1: {train_f1:.4f}"
            )
            if self.val_loader:
                status += f" || Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | F1: {val_f1:.4f}"
            print(status)

            # Step Learning Rate Scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    metric = val_loss if self.config.scheduler_monitor_metric == "val_loss" else train_loss
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()

            # Checkpoint management logic
            self.save_checkpoint(self.config.last_model_filename)
            if self.val_loader and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(self.config.best_model_filename)
                print(f"🏆 Best validation loss updated to {val_loss:.4f}. Model saved.")
            
            # Post-epoch global optimization cleanup
            self._clear_vram_cache()

        return self.history

    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """Plots the training metrics history (Loss, Accuracy, F1 Score)."""
        epochs = range(1, len(self.history["train_loss"]) + 1)
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        metrics_to_plot = [("loss", "Loss"), ("acc", "Accuracy"), ("f1", "F1 Score")]

        for i, (key, title) in enumerate(metrics_to_plot):
            axs[i].plot(epochs, self.history[f"train_{key}"], label=f"Train {title}", marker='o')
            if self.val_loader and f"val_{key}" in self.history:
                axs[i].plot(epochs, self.history[f"val_{key}"], label=f"Val {title}", marker='s')
            
            axs[i].set_title(title)
            axs[i].set_xlabel("Epochs")
            axs[i].set_ylabel(title)
            axs[i].legend()
            axs[i].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"📈 Evaluation plot successfully saved to: {save_path}")
        plt.show()

    def save_checkpoint(self, filename: str) -> None:
        """Saves current state and entire tracked training history metadata to disk."""
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename: str) -> None:
        """Restores training weights state and historical loss records from a saved file."""
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at path: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)
        print(f"Successfully restored state context parameters from: {filepath}")


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    apply_softmax: bool = False,
) -> torch.Tensor:
    """Generates continuous raw logits or probabilities from an evaluation inference pipeline."""
    model.eval()
    target_device = torch.device(device)
    model.to(target_device)
    all_predictions = []

    with torch.no_grad():
        for inputs in tqdm(data_loader, desc="Inference Phase", leave=False):
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]

            inputs = inputs.to(target_device)
            outputs = model(inputs)

            if apply_softmax:
                outputs = torch.softmax(outputs, dim=1)

            all_predictions.append(outputs.cpu())

    return torch.cat(all_predictions, dim=0)


def save_model_weights(model: nn.Module, filepath: str) -> None:
    """Isolates and serializes the model's raw learned weights parameter matrix."""
    torch.save(model.state_dict(), filepath)
    print(f"Model weight parameters explicitly written to: {filepath}")


def load_model_weights(model: nn.Module, filepath: str, device: str = "cpu") -> nn.Module:
    """Injects saved weights parameter tensors directly back into an active architecture."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Target weight file path missing: {filepath}")
    state_dict = torch.load(filepath, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    print(f"Successfully assigned model parameters out from: {filepath}")
    return model


def download_kaggle_dataset(dataset_name: str) -> str:
    """Downloads a Kaggle dataset to a local directory given its identifier, returning the local path."""
    import kagglehub
    from pathlib import Path
    path = Path(kagglehub.dataset_download(dataset_name))
    print(f"Path to the dataset: {path}")
    print(f"Is the path a file? {os.path.isfile(path)}")
    print(f"Files inside the path: {os.listdir(path)}")
    return str(path)