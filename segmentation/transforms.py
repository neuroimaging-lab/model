from monai.transforms import Compose, DivisiblePad

train_transforms = Compose(
    [
        DivisiblePad(
            k=16,
        ),
    ]
)
