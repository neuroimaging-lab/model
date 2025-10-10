from pathlib import Path
from typing import Any,Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from segmentation.config import DATA_LABELS
from util.image_manipulation import cut_images_and_masks
from segmentation.dataset import BrainTumorDataset
from segmentation.transforms import train_transforms

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
            modalities=["FLAIR", "T1w", "t1gd", "T2w"],
            transform=train_transforms,
            target_transform=train_transforms,
            cache_data=False,
        )

        final_dataset: Union[BrainTumorDataset, torch.utils.data.Subset[Any]]

        if max_total_samples > 0 and max_total_samples < len(full_dataset):
            final_dataset = torch.utils.data.Subset(
                full_dataset, range(max_total_samples)
            )
            total_samples = max_total_samples
        else:
            final_dataset = full_dataset
            total_samples = final_dataset.__sizeof__()

        print(f"Total samples in dataset: {total_samples}")
        val_size = int(val_split * total_samples)
        val_size = max(1, val_size) if total_samples > 1 else 0
        train_size = total_samples - val_size

        train_dataset = torch.utils.data.Subset(final_dataset, range(train_size))

        if val_size <= 0:
            print(
                "No validation samples available, using the same as for the training."
            )  # It is strictly for overfitting check on 1 sample
            val_dataset = train_dataset
        else:
            val_dataset = torch.utils.data.Subset(
                final_dataset, range(train_size, max_total_samples)
            )

        print(
            f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
        )
        return train_dataset, val_dataset

def create_dataloaders(
        data_config,
        train_dataset: torch.utils.data.Subset,
        val_dataset: torch.utils.data.Subset,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training and validation datasets to use during training."""
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

def ensure_class_indices(
        target: torch.Tensor, num_classes: int
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

def prepare_images_and_masks(images, masks, device):
    global ONCE
    images, masks = cut_images_and_masks(images, masks)

    if not ONCE:
        print_probability_of_each_class(masks[0])

    images = images.to(device, non_blocking=True)
    masks = masks.to(device, non_blocking=True)

    masks = ensure_class_indices(
        masks,
        num_classes=images.shape[1],
    )

    return images, masks

ONCE = False

def print_probability_of_each_class(example_image_mask: torch.Tensor):
    mask_np = example_image_mask.squeeze(0).cpu().numpy()
    unique_classes, counts = np.unique(mask_np, return_counts=True)

    total_voxels = mask_np.size
    proportions = {
        DATA_LABELS[int(cls)]: count / total_voxels
        for cls, count in zip(unique_classes, counts)
    }

    print("\nClass proportions in the selected MRI image:")
    for label, proportion in proportions.items():
        print(f"Class '{label}': {proportion:.7f}")
    
    global ONCE
    ONCE = True
