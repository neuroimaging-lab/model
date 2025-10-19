import torch

from segmentation.config import CLUSTER_COMPUTER_ENABLED, DATASET_DIR
from segmentation.models.unet3d import UNet3D
from segmentation.train import Trainer


def example_train():
    if CLUSTER_COMPUTER_ENABLED:
        model = UNet3D(
            in_channels=4,
            num_classes=4,
            level_channels=[64, 128, 256],
            bottleneck_channels=512,
        )
    else:
        model = UNet3D(
            in_channels=4,
            num_classes=4,
            level_channels=[64, 128, 256],
            bottleneck_channels=512,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size=4 if CLUSTER_COMPUTER_ENABLED else 1,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer.train(
        dataset_dir=DATASET_DIR,
        total_samples=20,
        epochs=10,
        device=device,
    )


if __name__ == "__main__":
    example_train()
