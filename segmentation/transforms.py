from monai.transforms import (
    Compose,
    DivisiblePadD,
    RandAffineD,
    RandFlipD,
    RandGaussianNoiseD,
    RandRotateD,
)

train_transforms = Compose(
    [
        DivisiblePadD(keys=["image", "label"], k=16),
        RandFlipD(keys=["image", "label"], prob=0.5, spatial_axis=[0]),
        RandFlipD(keys=["image", "label"], prob=0.5, spatial_axis=[1]),
        RandFlipD(keys=["image", "label"], prob=0.5, spatial_axis=[2]),
        RandRotateD(
            keys=["image", "label"], range_x=0.2, prob=0.3, mode=["bilinear", "nearest"]
        ),
        RandRotateD(
            keys=["image", "label"], range_y=0.2, prob=0.3, mode=["bilinear", "nearest"]
        ),
        RandRotateD(
            keys=["image", "label"], range_z=0.2, prob=0.3, mode=["bilinear", "nearest"]
        ),
        RandAffineD(
            keys=["image", "label"],
            prob=0.3,
            translate_range=(5, 5, 5),
            scale_range=(0.1, 0.1, 0.1),
            padding_mode="border",
            mode=["bilinear", "nearest"],
        ),
        RandGaussianNoiseD(
            keys=["image"],
            prob=0.2,
            std=0.05,
        ),
    ]
)


val_transforms = Compose(
    [
        DivisiblePadD(keys=["image", "label"], k=16),
    ]
)
