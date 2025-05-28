from pathlib import Path

import torch

from segmentation.dataset import BrainTumorDataset
from segmentation.transforms import train_transforms


def data_access():
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    dataset_dir = project_root / "datasets" / "Task01_BrainTumour"

    train_dataset = BrainTumorDataset(
        root_dir=str(dataset_dir),
        split="train",
        modalities=["FLAIR", "T1w", "t1gd", "T2w"],
    )

    print(f"Dataset size: {len(train_dataset)} samples")
    print(f"Label mapping: {train_dataset.get_label_mapping()}")

    image, mask = train_dataset[0]
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    if mask is not None:
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Unique labels: {torch.unique(mask)}")


def data_access_and_transform():
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    dataset_dir = project_root / "datasets" / "Task01_BrainTumour"

    train_dataset = BrainTumorDataset(
        root_dir=str(dataset_dir),
        split="train",
        modalities=["FLAIR", "T1w", "t1gd", "T2w"],
        transform=train_transforms,
        target_transform=train_transforms,
    )

    print(f"Dataset size: {len(train_dataset)} samples")
    print(f"Label mapping: {train_dataset.get_label_mapping()}")

    image, mask = train_dataset[0]
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    if mask is not None:
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
        print(f"Unique labels: {torch.unique(mask)}")


if __name__ == "__main__":
    data_access()
    data_access_and_transform()
