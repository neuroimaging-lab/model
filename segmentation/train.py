from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation.dataset import BrainTumorDataset
from segmentation.transforms import train_transforms


class Trainer:
    """
    Trainer class for training and validating a brain tumor segmentation model.
    Trains with multi-class Dice loss, and reports mean Dice (optionally excluding background).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        batch_size: int = 1,
        val_split: float = 0.2,
        save_dir: Path | str = "checkpoints",
        num_workers: int = 2,
        pin_memory: bool = True,
        include_bg: bool = False,  # Background (usually dominant) can negatively impact Dice score
    ) -> None:
        self.model = model
        self.optimizer = optimizer

        self.data_config: Dict[str, Any] = {
            "batch_size": batch_size,
            "val_split": val_split,
            "save_dir": Path(save_dir),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        self.include_bg = include_bg

    def train(
        self,
        dataset_dir: Path,
        total_samples: int = -1,
        epochs: int = 5,
        device: str = "cpu",
        checkpoints_enabled: bool = False,
    ) -> None:
        train_dataset, val_dataset = self._get_datasets(dataset_dir, total_samples)
        train_loader, val_loader = self._create_dataloaders(train_dataset, val_dataset)

        self.model = self.model.to(device)

        best_val_dice: float = 0.0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir: Path = self.data_config["save_dir"] / f"run_{timestamp}"
        save_dir.mkdir(exist_ok=True, parents=True)

        print(f"Training on device: {device}")

        start = time.time()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss, train_dice = self._train_one_epoch(train_loader, device)
            val_loss, val_dice = self._validate(val_loader, device)

            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"Val   Loss: {val_loss:.4f}, Val   Dice: {val_dice:.4f}")

            if checkpoints_enabled:
                self._save_model(
                    save_dir=save_dir,
                    epoch=epoch,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    val_dice=val_dice,
                )

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                self._save_model(
                    save_dir=save_dir,
                    epoch=epoch,
                    model_state_dict=self.model.state_dict(),
                    optimizer_state_dict=self.optimizer.state_dict(),
                    val_dice=val_dice,
                    is_best_model=True,
                )
                print(f"Saved best model with Dice score: {val_dice:.4f}")

        train_time = time.time() - start
        train_time = int(train_time)
        print(f"Training completed in {timedelta(seconds=train_time)}")

    def _get_datasets(
        self,
        root_dir: Path,
        total_samples: int,
    ) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
        """
        Build the full training dataset, optionally subsample, then split into train/val.

        Args:
            root_dir: The root directory of the dataset.
            total_samples: The total number of samples to use (-1 for all).

        Returns:
            Tuple containing the training and validation datasets.
        """
        full_dataset = BrainTumorDataset(
            root_dir=str(root_dir),
            split="train",
            modalities=["FLAIR", "T1w", "t1gd", "T2w"],
            transform=train_transforms,
            target_transform=train_transforms,
            cache_data=False,
        )

        final_dataset: torch.utils.data.Dataset

        if total_samples > 0 and total_samples < len(full_dataset):
            subset_size = total_samples
            subset = torch.utils.data.Subset(full_dataset, range(subset_size))
            final_dataset = subset
        else:
            final_dataset = full_dataset

        total_samples = final_dataset.__sizeof__()
        val_size = int(self.data_config["val_split"] * total_samples)
        val_size = max(1, val_size) if total_samples > 1 else 0
        train_size = total_samples - val_size

        if train_size <= 0 or val_size <= 0:
            raise ValueError(
                f"Insufficient samples for training and validation splits: {total_samples}"
            )

        train_dataset = torch.utils.data.Subset(final_dataset, range(train_size))

        val_dataset = torch.utils.data.Subset(
            final_dataset, range(train_size, total_samples)
        )

        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        return train_dataset, val_dataset

    def _create_dataloaders(
        self,
        train_dataset: torch.utils.data.Subset,
        val_dataset: torch.utils.data.Subset,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training and validation datasets to use during training."""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=True,
            num_workers=self.data_config["num_workers"],
            persistent_workers=True,
            pin_memory=self.data_config["pin_memory"],
        )  # We are shuffling the training data for better generalization

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config["num_workers"],
            persistent_workers=True,
            pin_memory=self.data_config["pin_memory"],
        )

        # The num_workers and pin_memory arguments are used to speed up data loading,
        # persistent_workers is used to avoid reinitializing those workers each epoch

        return train_loader, val_loader

    def _ensure_class_indices(
        self, target: torch.Tensor, num_classes: int
    ) -> torch.Tensor:
        """
        Ensure target is class indices tensor of shape (B, 1, D, H, W), dtype long.
        Accepts:
          - (B, 1, D, H, W) with ints
          - (B, C, D, H, W) one-hot (will be argmaxed)
          - (B, D, H, W) (will add channel dim)
        """
        # We want to keep the shape consistent here, (B, D, H, W), later ensure the values are of dtype long
        # and finally add channel dim to return tensor with shape (B, 1, D, H, W)
        if target.ndim == 5 and target.shape[1] == 1:
            tgt = target.squeeze(1)
        elif target.ndim == 5 and target.shape[1] == num_classes:
            tgt = torch.argmax(target, dim=1)
        elif target.ndim == 4:
            tgt = target
        else:
            raise ValueError(f"Unexpected target shape: {tuple(target.shape)}")

        if tgt.dtype != torch.long:
            tgt = tgt.long()

        return tgt.unsqueeze(1)

    def _dice_loss(
        self,
        logits: torch.Tensor,  # (B, C, D, H, W)
        target: torch.Tensor,  # (B, 1, D, H, W) containing class indices (dtype long)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Multi-class soft Dice loss (probabilities vs one-hot target)."""
        # This normalizes the logits (which are raw, unnormalized scores the model outputs)
        # across classes, so that they sum to 1 across classes
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)
        c = probs.shape[1]

        tgt = target.squeeze(1).long()  # (B, D, H, W)
        # Now we create vector instead of class index at each voxel, so if we had 4 classes, and at voxel (d, h, w) the class is 2,
        # the vector will be [0, 1, 0, 0], later we reorganize tensor shape to match (B, C, D, H, W)
        one_hot = F.one_hot(tgt, num_classes=c).permute(0, 4, 1, 2, 3).float()

        if not self.include_bg and c > 1:
            # If we do not include background, we remove the first channel (background has index 0) with slicing
            probs = probs[:, 1:, ...]
            one_hot = one_hot[:, 1:, ...]

        dims = (
            0,
            2,
            3,
            4,
        )  # Summing over batch (which is typically 1) and spatial dimensions for each class

        # In intersection section it is basically multiplying vectors like [0.1, 0.2, 0.5, 0.2] by [0, 1, 0, 0],
        # which results in [0, 0.2, 0, 0], then summing. This `soft` approach allows for partial credit,
        # instead of receiving 0 in that example as the highest probability class was not the true class.
        # This approach can help in gradient flow and learning.
        intersection = (probs * one_hot).sum(dim=dims)

        # In the denominator, we sum the probabilities and one-hot vectors
        denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)

        dice_per_class = (2.0 * intersection + eps) / (denom + eps)

        return 1.0 - dice_per_class.mean()

    def _dice_coefficient(
        self,
        logits: torch.Tensor,  # (B, C, D, H, W)
        target: torch.Tensor,  # (B, 1, D, H, W) containing class indices (int)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Multi-class Dice on argmax predictions. Returns mean Dice across classes."""
        preds = torch.argmax(logits, dim=1)  # (B, D, H, W)
        c = logits.shape[1]
        tgt = target.squeeze(1).long()  # (B, D, H, W)

        pred = F.one_hot(preds, num_classes=c).permute(0, 4, 1, 2, 3).float()
        tgt = F.one_hot(tgt, num_classes=c).permute(0, 4, 1, 2, 3).float()

        if not self.include_bg and c > 1:
            pred = pred[:, 1:, ...]
            tgt = tgt[:, 1:, ...]

        dims = (0, 2, 3, 4)

        # Here we no longer have floats in pred vectors, now we use argmax, so we have hard predictions
        # and the only way to get non 0 value, is if the class is the same in both pred and tgt
        intersection = (pred * tgt).sum(dim=dims)
        denom = pred.sum(dim=dims) + tgt.sum(dim=dims)

        dice_per_class = (2.0 * intersection + eps) / (denom + eps)

        return dice_per_class.mean()

    def _train_one_epoch(
        self,
        dataloader: DataLoader,
        device: str,
    ) -> Tuple[float, float]:
        self.model.train()
        epoch_loss: float = 0.0
        dice_scores: List[float] = []

        with tqdm(dataloader, desc="Training") as progress:
            for _, (images, masks) in enumerate(progress):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                masks = self._ensure_class_indices(
                    masks,
                    num_classes=images.shape[1],
                )

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)

                loss = self._dice_loss(outputs, masks)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    dice = self._dice_coefficient(outputs, masks)
                    dice_scores.append(dice.item())
                    epoch_loss += loss.item()

                progress.set_postfix(
                    {"loss": f"{loss.item():.4f}", "dice": f"{dice_scores[-1]:.4f}"}
                )

        return epoch_loss / max(1, len(dataloader)), float(
            np.mean(dice_scores) if dice_scores else 0.0
        )

    def _validate(
        self,
        dataloader: DataLoader,
        device: str,
    ) -> Tuple[float, float]:
        self.model.eval()
        val_loss: float = 0.0
        dice_scores: List[float] = []

        with torch.no_grad():
            with tqdm(dataloader, desc="Validation") as progress:
                for _, (images, masks) in enumerate(progress):
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)

                    masks = self._ensure_class_indices(
                        masks,
                        num_classes=images.shape[1],
                    )

                    outputs = self.model(images)
                    loss = self._dice_loss(outputs, masks)

                    dice = self._dice_coefficient(outputs, masks)
                    dice_scores.append(dice.item())
                    val_loss += loss.item()

                    progress.set_postfix(
                        {
                            "val_loss": f"{loss.item():.4f}",
                            "val_dice": f"{dice_scores[-1]:.4f}",
                        }
                    )

        return val_loss / max(1, len(dataloader)), float(
            np.mean(dice_scores) if dice_scores else 0.0
        )

    def _save_model(
        self,
        save_dir: Path,
        epoch: int,
        model_state_dict: dict[str, Any],
        optimizer_state_dict: dict[str, Any],
        val_dice: float,
        is_best_model: bool = False,
    ) -> None:
        """Save the model and optimizer state dictionaries to a file."""
        filepath = (
            save_dir / f"epoch_{epoch}.pth"
            if not is_best_model
            else save_dir / "best_model.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "val_dice": val_dice,
            },
            filepath,
        )
