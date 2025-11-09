import copy
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import torch
from torch.utils.data import DataLoader, Subset as TorchSubset

from segmentation.config import CLUSTER_TRAINING_ENABLED
from segmentation.dataset import BrainTumorDataset
from segmentation.transforms import train_transforms, val_transforms


def get_datasets(
    val_split,
    root_dir: Path,
    max_total_samples: int,
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
    )

    if max_total_samples > 0 and max_total_samples < len(full_dataset):
        total_samples = max_total_samples
    else:
        total_samples = len(full_dataset)

    print(f"Total samples in dataset: {total_samples}")
    val_size = int(val_split * total_samples)
    val_size = max(1, val_size) if total_samples > 1 else 0
    train_size = total_samples - val_size

    indices = list(range(total_samples))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:total_samples]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(copy.deepcopy(full_dataset), val_indices)

    train_inner = cast(BrainTumorDataset, train_dataset.dataset)
    val_inner = cast(BrainTumorDataset, val_dataset.dataset)

    train_inner.transform = train_transforms
    train_inner.target_transform = train_transforms

    val_inner.transform = val_transforms
    val_inner.target_transform = val_transforms

    print("Applied transforms:")
    print(train_inner.transform)
    print(val_inner.transform)
    print("-------------------------")
    indices = list(range(total_samples))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:total_samples]

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(copy.deepcopy(full_dataset), val_indices)

    train_inner = cast(BrainTumorDataset, train_dataset.dataset)
    val_inner = cast(BrainTumorDataset, val_dataset.dataset)

    train_inner.transform = train_transforms
    train_inner.target_transform = train_transforms

    val_inner.transform = val_transforms
    val_inner.target_transform = val_transforms

    print("Applied transforms:")
    print(train_inner.transform)
    print(val_inner.transform)
    print("-------------------------")

    if val_size <= 0:
        print(
            "No validation samples available, using the same as for the training."
        )  # It is strictly for overfitting check on 1 sample
        val_dataset = train_dataset

    print(
        f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
    )

    train_ids = get_sample_ids(train_dataset)
    val_ids = get_sample_ids(val_dataset)

    overlap = set(train_ids) & set(val_ids)

    print(f"\nTrain unique samples: {len(set(train_ids))}")
    print(f"Val unique samples:   {len(set(val_ids))}")
    print(f"Overlap count:        {len(overlap)}")

    return train_dataset, val_dataset


def get_sample_ids(subset: TorchSubset[BrainTumorDataset]) -> list[str]:
    ds = cast(BrainTumorDataset, subset.dataset)
    indices = subset.indices

    return [ds.file_list[idx]["image"] for idx in indices]


def create_dataloaders(
    data_config: Dict[str, Any], root_dir: Path, max_total_samples: int
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation datasets to use during training."""

    train_dataset, val_dataset = get_datasets(
        data_config["val_split"], root_dir, max_total_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        persistent_workers=True,
        pin_memory=data_config["pin_memory"],
    )  # We are shuffling the training data for better generalization

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        persistent_workers=True,
        pin_memory=data_config["pin_memory"],
    )

    # The num_workers and pin_memory arguments are used to speed up data loading,
    # persistent_workers is used to avoid reinitializing those workers each epoch

    return train_loader, val_loader


def save_model(
    save_dir: Path,
    epoch: int,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any],
    val_dice: float,
) -> None:
    """Save the model and optimizer state dictionaries to a file."""
    filepath = save_dir / "best_model.pth"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "val_dice": val_dice,
        },
        filepath,
    )


def cut_images_and_masks(
    images: torch.Tensor, masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reduces voxel dimensions for the images and corresponding masks, by center-cropping.
    Used as a test tool for the local development - (e.g. 160x240x240 → 80x160x160) - to be able to fit an MRI image inside a model with full channels.
    """
    depth = 80  # base 160
    height = 160  # base 240
    width = 160  # base 240

    start_d = (160 - depth) // 2
    start_h = (240 - height) // 2
    start_w = (240 - width) // 2

    return images[
        :,
        :,
        start_d : start_d + depth,
        start_h : start_h + height,
        start_w : start_w + width,
    ], masks[
        :,
        :,
        start_d : start_d + depth,
        start_h : start_h + height,
        start_w : start_w + width,
    ]


def prepare_images_and_masks(
    images: torch.Tensor, masks: torch.Tensor, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not CLUSTER_TRAINING_ENABLED:
        images, masks = cut_images_and_masks(images, masks)

    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)

    return images, masks
