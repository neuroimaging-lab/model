from datetime import datetime
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from segmentation.dataset import BrainTumorDataset
from segmentation.transforms import train_transforms


class Trainer:
    """
    Trainer class for training and validating a brain tumor segmentation model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.data_config: Dict[str, Any] = {
            "batch_size": 1,
            "val_split": 0.2,
            "save_dir": Path("checkpoints"),
        }
        self.model = model
        self.optimizer = optimizer

    def get_datasets(
        self,
        root_dir: Path,
        total_samples: int,
    ) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
        full_dataset = BrainTumorDataset(
            root_dir=str(root_dir),
            split="train",
            modalities=["FLAIR", "T1w", "t1gd", "T2w"],
            transform=train_transforms,
            target_transform=train_transforms,
            cache_data=False,
        )

        final_dataset: Dataset

        if total_samples > 0 and total_samples < len(full_dataset):
            indices = torch.randperm(len(full_dataset))[:total_samples].tolist()
            final_dataset = torch.utils.data.Subset(full_dataset, indices)
        else:
            final_dataset = full_dataset
            total_samples = len(full_dataset)

        val_size = int(self.data_config["val_split"] * total_samples)
        train_size = total_samples - val_size

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
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.data_config["batch_size"],
            shuffle=False,
        )

        return train_loader, val_loader

    def dice_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        pred = pred.contiguous()
        target = target.contiguous()

        # tensor shape: (batch_size, num_classes, depth, height, width)
        intersection = (pred * target).sum(dim=(2, 3, 4))
        loss = 1 - (
            (2 * intersection) / (pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)))
        )

        return loss.mean()

    def dice_coefficient(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        pred = (pred > threshold).float()
        intersection = (pred * target).sum()

        return (2 * intersection) / (pred.sum() + target.sum())

    def train_one_epoch(
        self,
        dataloader: DataLoader,
        device: str,
    ) -> tuple[float, float]:
        self.model.train()
        epoch_loss: float = 0.0
        dice_scores: List[float] = []

        with tqdm(dataloader, desc="Training") as progress:
            for _, (images, masks) in enumerate(progress):
                images, masks = images.to(device), masks.to(device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.dice_loss(outputs, masks)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    dice = self.dice_coefficient(outputs, masks)
                    dice_scores.append(dice.item())
                    epoch_loss += loss.item()

        return epoch_loss / len(dataloader), float(np.mean(dice_scores))

    def validate(
        self,
        dataloader: DataLoader,
        device: str,
    ) -> tuple[float, float]:
        self.model.eval()
        val_loss: float = 0.0
        dice_scores: List[float] = []

        with torch.no_grad():
            with tqdm(dataloader, desc="Validation") as progress:
                for _, (images, masks) in enumerate(progress):
                    images, masks = images.to(device), masks.to(device)

                    outputs = self.model(images)
                    loss = self.dice_loss(outputs, masks)

                    dice = self.dice_coefficient(outputs, masks)
                    dice_scores.append(dice.item())
                    val_loss += loss.item()

        return val_loss / len(dataloader), float(np.mean(dice_scores))

    def train(
        self,
        dataset_dir: Path,
        total_samples: int = -1,
        epochs: int = 5,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        train_dataset, val_dataset = self.get_datasets(dataset_dir, total_samples)
        train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)

        self.model = self.model.to(device)

        best_val_dice: float = 0.0

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self.data_config["save_dir"] / f"run_{timestamp}"
        save_dir.mkdir(exist_ok=True, parents=True)

        train_losses: List[float] = []
        val_losses: List[float] = []
        train_dices: List[float] = []
        val_dices: List[float] = []

        print(f"Training on device: {device}")

        start = time.time()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss, train_dice = self.train_one_epoch(train_loader, device)
            train_losses.append(train_loss)
            train_dices.append(train_dice)

            val_loss, val_dice = self.validate(val_loader, device)
            val_losses.append(val_loss)
            val_dices.append(val_dice)

            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

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

        print(
            f"Training completed in {train_time:.2f} seconds ({train_time / 60:.2f} minutes)"
        )

        return {
            "model": self.model,
            "best_dice": best_val_dice,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_dices": train_dices,
            "val_dices": val_dices,
            "train_time": train_time,
        }
