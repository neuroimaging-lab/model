from argparse import ArgumentParser
from typing import Optional

import torch

from segmentation.config import CLUSTER_TRAINING_ENABLED, DATASET_DIR
from segmentation.models.unet3d import UNet3D
from segmentation.train import Trainer


def example_train(gamma: Optional[float], alpha: Optional[float]):
    if CLUSTER_TRAINING_ENABLED:
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
        batch_size=4 if CLUSTER_TRAINING_ENABLED else 1,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer.train(
        dataset_dir=DATASET_DIR,
        total_samples=20,
        epochs=10,
        device=device,
        gamma=gamma,
        alpha=alpha,
    )


def parse_args() -> tuple[Optional[float], Optional[float]]:
    parser: ArgumentParser = ArgumentParser(description="Fine tunning runner")
    parser.add_argument("--gamma", nargs="+", type=float, help="Gamma value")
    parser.add_argument("--alpha", nargs="+", type=float, help="Alpha value")

    try:
        args = parser.parse_args()
        return args.gamma[0], args.alpha[0]
    except Exception:
        return None, None


if __name__ == "__main__":
    example_train(*parse_args())
