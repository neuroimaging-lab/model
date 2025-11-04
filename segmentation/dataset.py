import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import nibabel as nib
from nibabel.spatialimages import SpatialImage
import numpy as np
import torch
from torch.utils.data import Dataset


class BrainTumorDataset(Dataset):
    """
    Dataset class for loading and processing BRATS brain tumor segmentation data.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        modalities: List[str] = ["FLAIR", "T1w", "t1gd", "T2w"],
        cache_data: bool = False,
        preload: bool = False,
    ) -> None:
        """
        Initialize the BrainTumorDataset.

        Args:
            root_dir: Root directory containing the dataset
            split: 'train' or 'test' split
            transform: Transformations to apply to input images
            target_transform: Transformations to apply to segmentation masks
            modalities: List of MRI modalities to include
            cache_data: Whether to cache data in memory after first access
            preload: Whether to preload all data into memory at initialization
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.cache_data = cache_data
        self.data_cache: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}

        with open(self.root_dir / "dataset.json", "r") as f:
            self.dataset_info = json.load(f)

        self.modality_map = {
            v: int(k) for k, v in self.dataset_info["modality"].items()
        }
        self.num_modalities = len(self.dataset_info["modality"])

        self.modalities = [
            self.modality_map[mod] for mod in modalities if mod in self.modality_map
        ]

        if split == "train":
            self.file_list = self.dataset_info["training"]
        elif split == "test":
            self.file_list = [
                {"image": img_path, "label": None}
                for img_path in self.dataset_info["test"]
            ]
        else:
            raise ValueError(f"Split '{split}' not recognized. Use 'train' or 'test'.")

        # Preload data if requested
        if preload:
            print(f"Preloading {split} data into memory...")
            for idx in range(len(self.file_list)):
                self._load_and_cache_item(idx)
            print("Preloading complete.")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to fetch

        Returns:
            Tuple containing (image, mask) where image has shape [modalities, D, H, W]
            and mask has shape [1, D, H, W]. For test split, mask may be None.
        """
        if self.cache_data and idx in self.data_cache:
            image, mask = self.data_cache[idx]
        else:
            image, mask = self._load_and_cache_item(idx)

        if self.transform is not None:
            augmented = self.transform({"image": image, "label": mask})
            image, mask = augmented["image"], augmented["label"]

        if mask is not None:
            mask = mask.to(torch.int64)

        return image, mask

    def _load_and_cache_item(
        self, idx: int
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Load sample and optionally cache it in memory."""
        sample_info = self.file_list[idx]

        image_path = os.path.join(self.root_dir, sample_info["image"].lstrip("./"))
        nii_img = cast(
            SpatialImage, nib.load(image_path)
        )  # TODO: assuming that type is correct to silence mypy
        image_data = nii_img.get_fdata()  # Shape: [H, W, D, C]

        if len(self.modalities) > 0:
            image_data = image_data[..., self.modalities]

        # Transpose to [C, D, H, W] format for PyTorch
        image_data = np.transpose(image_data, (3, 2, 0, 1))
        image_tensor = torch.from_numpy(image_data).float()

        mask_tensor = None
        if sample_info["label"] is not None:
            mask_path = os.path.join(self.root_dir, sample_info["label"].lstrip("./"))

            nii_mask = cast(
                SpatialImage, nib.load(mask_path)
            )  # TODO: assuming that type is correct to silence mypy
            mask_data = nii_mask.get_fdata()  # Shape: [H, W, D]

            # Transpose to [1, D, H, W] format for PyTorch
            mask_data = np.expand_dims(np.transpose(mask_data, (2, 0, 1)), 0)
            mask_tensor = torch.from_numpy(mask_data).long()

        if self.cache_data:
            self.data_cache[idx] = (image_tensor, mask_tensor)

        return image_tensor, mask_tensor

    def get_label_mapping(self) -> Dict[int, str]:
        """Return the mapping from label indices to their descriptions."""
        return {int(k): v for k, v in self.dataset_info["labels"].items()}
