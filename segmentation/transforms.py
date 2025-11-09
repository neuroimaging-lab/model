from monai.transforms import (
    CastToTyped,
    Compose,
    CropForegroundd,
    DivisiblePadD,
    NormalizeIntensityd,
    RandAffined,
    RandBiasFieldd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandShiftIntensityd,
    SpatialPadd,
)
import torch

from segmentation.config import CLUSTER_TRAINING_ENABLED

train_transforms = Compose(
    [
        # Normalize input based on the mean and standard deviation of non-zero voxels
        #  Z-score (x - mean) / std per channel
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        # Remove background by cropping to the non-zero bounding box of the image
        CropForegroundd(keys=["image", "label"], source_key="image"),
        # Ensure the cropped volume is at least (128, 128, 128) patch size
        SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
        # Randomly pick one (128, 128, 128) patch (num_samples=1)
        #   pos: patch center is drawn from voxels with positive labels
        #   neg: patch center is drawn from voxels with zeros
        #   for pos=1 and neg=1, half patches will have tumor, half will be random
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=1,
            neg=1,
            num_samples=1,
        ),
        # Random flips along axes probability 0.5 to activate
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1]),
        # Random transformations: rotation, translation, scaling
        RandAffined(
            keys=["image", "label"],
            prob=0.3,  # probability to apply the transform
            rotate_range=(0.1, 0.1, 0.0),  # radians
            translate_range=(5, 5, 0),  # movements in voxels
            scale_range=(0.05, 0.05, 0.0),  # scaling factors
            padding_mode="border",  # after transform, we need to define how to fill empty voxels out of pov, filled by nearest neighbor values
            mode=(
                "bilinear",
                "nearest",
            ),  # bilinear smooths by averaging neighbors, nearest neighbor to keep integer classes
        ),
        # Random intensity augmentations
        RandBiasFieldd(keys=["image"], prob=0.3, coeff_range=(0.0, 0.05)),
        # Random intensity shift
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.2),
        # Random Gaussian noise, it improves robustness to noisy images
        RandGaussianNoised(keys=["image"], prob=0.15, std=0.01),
        # Guarantee label tensor dtype for losses/metrics
        CastToTyped(keys=["label"], dtype=torch.int64),
        # Pad to multiples of 16 (UNet down/upsampling compatibility)
        DivisiblePadD(keys=["image", "label"], k=16),
    ]
)

if CLUSTER_TRAINING_ENABLED:
    val_transforms = Compose(
        [
            # Same standardization as in training
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # Guarantee label tensor dtype for losses/metrics
            CastToTyped(keys=["label"], dtype=torch.int64),
            # Pad to multiples of 16 (UNet down/upsampling compatibility)
            DivisiblePadD(keys=["image", "label"], k=16),
        ]
    )
else:
    val_transforms = Compose(
        [
            # Same standardization as in training
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # Ensure the cropped volume is at least (128, 128, 128) patch size
            SpatialPadd(keys=["image", "label"], spatial_size=(128, 128, 128)),
            # Guarantee label tensor dtype for losses/metrics
            CastToTyped(keys=["label"], dtype=torch.int64),
            # Pad to multiples of 16 (UNet down/upsampling compatibility)
            DivisiblePadD(keys=["image", "label"], k=16),
        ]
    )
