from datetime import datetime
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation.dataset import BrainTumorDataset
from segmentation.transforms import train_transforms

CONFIG: Dict[str, Any] = {
    "batch_size": 1,
    "epochs": 5,
    "val_split": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": Path("checkpoints"),
}


def get_datasets(
    root_dir: Path,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    full_dataset = BrainTumorDataset(
        root_dir=str(root_dir),
        split="train",
        modalities=["FLAIR", "T1w", "t1gd", "T2w"],
        transform=train_transforms,
        target_transform=train_transforms,
        cache_data=False,
    )

    dataset_size = len(full_dataset)
    val_size = int(CONFIG["val_split"] * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"Training samples: {train_size}, Validation samples: {val_size}")

    return torch.utils.data.Subset(train_dataset, [0, 1, 2]), torch.utils.data.Subset(
        val_dataset, [0, 1]
    )
    # return train_dataset, val_dataset


def create_dataloaders(
    train_dataset: torch.utils.data.Subset, val_dataset: torch.utils.data.Subset
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )

    return train_loader, val_loader


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.contiguous()
    target = target.contiguous()

    # tensor shape: (batch_size, num_classes, depth, height, width)
    intersection = (pred * target).sum(dim=(2, 3, 4))
    loss = 1 - (
        (2 * intersection) / (pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4)))
    )

    return loss.mean()


def dice_coefficient(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()

    return (2 * intersection) / (pred.sum() + target.sum())


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
) -> tuple[float, float]:
    model.train()
    epoch_loss: float = 0.0
    dice_scores: List[float] = []

    with tqdm(dataloader, desc="Training") as progress:
        for _, (images, masks) in enumerate(progress):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                dice = dice_coefficient(outputs, masks)
                dice_scores.append(dice.item())
                epoch_loss += loss.item()

            if device == "cuda":
                torch.cuda.empty_cache()

    return epoch_loss / len(dataloader), float(np.mean(dice_scores))


def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str,
) -> tuple[float, float]:
    model.eval()
    val_loss: float = 0.0
    dice_scores: List[float] = []

    with torch.no_grad():
        with tqdm(dataloader, desc="Validation") as progress:
            for _, (images, masks) in enumerate(progress):
                images, masks = images.to(device), masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                dice = dice_coefficient(outputs, masks)
                dice_scores.append(dice.item())
                val_loss += loss.item()

    return val_loss / len(dataloader), float(np.mean(dice_scores))


def train(
    model: torch.nn.Module,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Any]:
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    dataset_dir = project_root / "datasets" / "Task01_BrainTumour"

    train_dataset, val_dataset = get_datasets(dataset_dir)
    train_loader, val_loader = create_dataloaders(train_dataset, val_dataset)

    device = CONFIG["device"]
    model = model.to(device)

    best_val_dice: float = 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = CONFIG["save_dir"] / f"run_{timestamp}"
    save_dir.mkdir(exist_ok=True, parents=True)

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_dices: List[float] = []
    val_dices: List[float] = []

    print(f"Training on device: {CONFIG['device']}")

    start = time.time()

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_losses.append(train_loss)
        train_dices.append(train_dice)

        val_loss, val_dice = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
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
        "model": model,
        "best_dice": best_val_dice,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_dices": train_dices,
        "val_dices": val_dices,
        "train_time": train_time,
    }
