from pathlib import Path

import torch

from segmentation.config import SUPER_COMPUTER_ENABLED
from segmentation.models.unet3d import UNet3D
from segmentation.train import Trainer


def example_train():
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    dataset_dir = project_root / "datasets" / "Task01_BrainTumour"

    if SUPER_COMPUTER_ENABLED:
        model = UNet3D(
            in_channels=4,
            num_classes=4,
        )
    else:
        model = UNet3D(
            in_channels=4,
            num_classes=4,
            level_channels=[32, 64, 128],
            bottleneck_channels=256,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer.train(
        dataset_dir=dataset_dir,
        total_samples=100,
        epochs=500,
        device=device,
        checkpoints_enabled=False,
    )


if __name__ == "__main__":
    example_train()
