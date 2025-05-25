from monai.transforms import DivisiblePad, Compose



train_transforms = Compose(
    [
        DivisiblePad(
            k=16,
        ),
    ]
)
