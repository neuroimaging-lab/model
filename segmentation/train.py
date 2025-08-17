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
    Trains with multi-class Dice loss (no CE), and reports mean Dice (optionally excluding background).
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
        include_bg: bool = False,  # whether to include background in Dice loss/metric
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

    def get_datasets(
        self,
        root_dir: Path,
        total_samples: int,
    ) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
        """
        Build the full training dataset from disk, optionally subsample, then split into train/val.
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
            remainder = len(full_dataset) - subset_size
            subset, _ = torch.utils.data.random_split(
                full_dataset, [subset_size, remainder]
            )
            final_dataset = subset
            total_samples = subset_size
        else:
            final_dataset = full_dataset
            total_samples = len(full_dataset)

        val_size = int(self.data_config["val_split"] * total_samples)
        val_size = max(1, val_size) if total_samples > 1 else 0
        train_size = total_samples - val_size

        # Ensure at least one training sample if possible
        if train_size == 0 and total_samples > 0:
            train_size, val_size = 1, total_samples - 1

        train_dataset, val_dataset = torch.utils.data.random_split(
            final_dataset, [train_size, val_size]
        )

        print(f"Training samples: {train_size}, Validation samples: {val_size}")
        return train_dataset, val_dataset

    def create_dataloaders(
        self,
        train_dataset: torch.utils.data.Subset,
        val_dataset: torch.utils.data.Subset,
    ) -> Tuple[DataLoader, DataLoader]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=True,
            num_workers=self.data_config["num_workers"],
            pin_memory=self.data_config["pin_memory"],
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
            num_workers=self.data_config["num_workers"],
            pin_memory=self.data_config["pin_memory"],
        )

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

    def dice_loss(
        self,
        logits: torch.Tensor,  # (B, C, D, H, W)
        target: torch.Tensor,  # (B, 1, D, H, W) containing class indices (int)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Multi-class soft Dice loss (probabilities vs one-hot target).
        Background is excluded by default (self.include_bg = False).
        """
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)
        c = probs.shape[1]

        tgt = target.squeeze(1).long()  # (B, D, H, W)
        one_hot = F.one_hot(tgt, num_classes=c).permute(0, 4, 1, 2, 3).float()

        if not self.include_bg and c > 1:
            probs = probs[:, 1:, ...]
            one_hot = one_hot[:, 1:, ...]

        dims = (0, 2, 3, 4)  # sum over batch + spatial dims, per-class
        intersection = (probs * one_hot).sum(dim=dims)
        denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice_per_class = (2.0 * intersection + eps) / (denom + eps)
        return 1.0 - dice_per_class.mean()

    def dice_coefficient(
        self,
        logits: torch.Tensor,  # (B, C, D, H, W)
        target: torch.Tensor,  # (B, 1, D, H, W) containing class indices (int)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Multi-class Dice on argmax predictions. Returns mean Dice across classes.
        Background is excluded by default (self.include_bg = False).
        """
        preds = torch.argmax(logits, dim=1)  # (B, D, H, W)
        c = logits.shape[1]
        tgt = target.squeeze(1).long()  # (B, D, H, W)

        pred_oh = F.one_hot(preds, num_classes=c).permute(0, 4, 1, 2, 3).float()
        tgt_oh = F.one_hot(tgt, num_classes=c).permute(0, 4, 1, 2, 3).float()

        if not self.include_bg and c > 1:
            pred_oh = pred_oh[:, 1:, ...]
            tgt_oh = tgt_oh[:, 1:, ...]

        dims = (0, 2, 3, 4)
        inter = (pred_oh * tgt_oh).sum(dim=dims)
        denom = pred_oh.sum(dim=dims) + tgt_oh.sum(dim=dims)
        dice_per_class = (2.0 * inter + eps) / (denom + eps)
        return dice_per_class.mean()

    def train_one_epoch(
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
                    num_classes=self.model.num_classes
                    if hasattr(self.model, "num_classes")
                    else images.shape[1],
                )

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)

                loss = self.dice_loss(outputs, masks)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    dice = self.dice_coefficient(outputs, masks)
                    dice_scores.append(dice.item())
                    epoch_loss += loss.item()

                progress.set_postfix(
                    {"loss": f"{loss.item():.4f}", "dice": f"{dice_scores[-1]:.4f}"}
                )

        return epoch_loss / max(1, len(dataloader)), float(
            np.mean(dice_scores) if dice_scores else 0.0
        )

    def validate(
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
                        num_classes=self.model.num_classes
                        if hasattr(self.model, "num_classes")
                        else images.shape[1],
                    )

                    outputs = self.model(images)
                    loss = self.dice_loss(outputs, masks)

                    dice = self.dice_coefficient(outputs, masks)
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

    def train(
        self,
        dataset_dir: Path,
        total_samples: int = -1,
        epochs: int = 5,
        device: str = "cpu",
        checkpoints_enabled: bool = False,
    ) -> Dict[str, Any]:
        train_dataset, val_dataset = self.get_datasets(dataset_dir, total_samples)
        train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)

        self.model = self.model.to(device)

        best_val_dice: float = 0.0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.data_config["save_dir"] / f"run_{timestamp}"
        save_dir.mkdir(exist_ok=True, parents=True)

        print(f"Training on device: {device}")

        start = time.time()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss, train_dice = self.train_one_epoch(train_loader, device)
            val_loss, val_dice = self.validate(val_loader, device)

            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"Val   Loss: {val_loss:.4f}, Val   Dice: {val_dice:.4f}")

            if checkpoints_enabled:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_dice": val_dice,
                    },
                    save_dir / f"epoch_{epoch}.pth",
                )

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_dice": val_dice,
                    },
                    save_dir / "best_model.pth",
                )
                print(f"Saved best model with Dice score: {val_dice:.4f}")

        train_time = time.time() - start
        train_time = int(train_time)
        print(f"Training completed in {timedelta(seconds=train_time)}")

        return {
            "best_val_dice": best_val_dice,
            "save_dir": str(save_dir),
            "epochs": epochs,
            "train_time_sec": train_time,
        }
