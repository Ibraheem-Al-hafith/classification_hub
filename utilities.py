"""Utility module for training, evaluating, and checkpointing PyTorch models.

This module provides a generic, reusable framework for binary and multi-class
classification tasks following the DRY principle. It decouples configuration,
optimization routines, and training loops from specific datasets or architectures.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
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
    
    # Track metrics for scheduler choices (e.g., ReduceLROnPlateau)
    scheduler_monitor_metric: str = "val_loss"

    # Checkpointing & Saving paths
    checkpoint_dir: str = "./checkpoints"
    best_model_filename: str = "best_model.pt"
    last_model_filename: str = "last_model.pt"

    # Optional kwargs for schedulers passed dynamically
    scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)


class Trainer:
    """A generic PyTorch Trainer encapsulated to handle multi-class and binary classification loops."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        """Initializes the Trainer with model, config, loss, and data loaders.

        Args:
            model: The PyTorch neural network model to train.
            config: An instance of TrainerConfig containing hyperparameters.
            criterion: The loss function (e.g., nn.CrossEntropyLoss, nn.BCEWithLogitsLoss).
            train_loader: DataLoader containing the training dataset.
            val_loader: Optional DataLoader containing the validation dataset.
        """
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

        # Initialize Scheduler via Factory Pattern
        self.scheduler = self._get_scheduler()

        # Internal state tracking
        self.best_val_loss = float("inf")
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
            raise ValueError(
                f"Unknown scheduler type: '{self.config.scheduler_type}'. "
                f"Supported choices: {list(scheduler_map.keys())}"
            )

        return scheduler_map[normalized_type](self.optimizer)

    def _run_epoch(self, is_train: bool = True) -> float:
        """Runs a single epoch of training or validation.

        Args:
            is_train: Flag to toggle training mode vs evaluation mode.

        Returns:
            The average calculated loss over the entire epoch.
        """
        self.model.train(is_train)
        loader = self.train_loader if is_train else self.val_loader
        
        if loader is None:
            return 0.0

        running_loss = 0.0
        desc = "Training" if is_train else "Validation"

        # Explicitly declare torch.set_grad_enabled context block
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
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return running_loss / len(loader.dataset)

    def fit(self) -> Dict[str, list[float]]:
        """Executes the complete training and evaluation orchestration lifecycle.

        Returns:
            A dictionary containing historical logs of training and validation losses.
        """
        history: Dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        print(f"Starting training loop on device: {self.device}")
        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(is_train=True)
            history["train_loss"].append(train_loss)

            val_loss = 0.0
            if self.val_loader:
                val_loss = self._run_epoch(is_train=False)
                history["val_loss"].append(val_loss)

            # Print concise progress summaries
            status_msg = f"Epoch [{epoch}/{self.config.epochs}] | Train Loss: {train_loss:.4f}"
            if self.val_loader:
                status_msg += f" | Val Loss: {val_loss:.4f}"
            print(status_msg)

            # Step learning rate scheduler based on its signature requirements
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    metric_to_monitor = val_loss if self.config.scheduler_monitor_metric == "val_loss" else train_loss
                    self.scheduler.step(metric_to_monitor)
                else:
                    self.scheduler.step()

            # Handle Model Checkpointing
            self.save_checkpoint(self.config.last_model_filename)
            if self.val_loader and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(self.config.best_model_filename)
                print(f"🏆 New best validation loss recorded ({val_loss:.4f}). Checkpoint saved.")

        return history

    def save_checkpoint(self, filename: str) -> None:
        """Saves a comprehensive execution state dictionary checkpoint to disk."""
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename: str) -> None:
        """Restores training and weights state context dynamically from a saved file."""
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at path: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Successfully restored state context parameters from: {filepath}")


def predict(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    apply_softmax: bool = False,
) -> torch.Tensor:
    """Generates continuous raw logits or probabilities from an evaluation inference pipeline.

    Args:
        model: Trained PyTorch architecture module.
        data_loader: DataLoader containing target dataset instances.
        device: Active system hardware computation environment.
        apply_softmax: Converts logits into explicit percentage probabilities if Multi-Class.

    Returns:
        A unified multi-dimensional tensor stacking overall prediction array batches.
    """
    model.eval()
    target_device = torch.device(device)
    model.to(target_device)
    all_predictions = []

    with torch.no_grad():
        for inputs in tqdm(data_loader, desc="Inference Phase", leave=False):
            # Workaround if DataLoader packs targets alongside input batches during inference calls
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
    """
    download kaggle dataset to local directory given the dataset identifier, return the dataset path
    """
    import kagglehub
    from pathlib import Path
    path = Path(kagglehub.dataset_download(dataset_name))
    print(f"Path to the dataset: {path}")
    print(f"Is the path a file? {os.path.isfile(path)}")
    print(f"Files inside the path: {os.listdir(path)}")
    return str(path)