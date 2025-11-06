from argparse import ArgumentParser
from typing import Optional

import torch

from segmentation.config import CLUSTER_TRAINING_ENABLED, DATASET_DIR
from segmentation.models.unet3d import UNet3D
from segmentation.train import Trainer


def example_train(
    gamma: Optional[float],
    alpha: Optional[float],
    lr: Optional[float],
    store_checkpoints_in_temp_storage: bool,
):
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr if lr else 3e-4)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        batch_size=4 if CLUSTER_TRAINING_ENABLED else 1,
        store_checkpoints_in_temp_storage=store_checkpoints_in_temp_storage,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer.train(
        dataset_dir=DATASET_DIR,
        total_samples=5,
        epochs=2,
        device=device,
        gamma=gamma,
        alpha=alpha,
    )


def parse_args() -> tuple[Optional[float], Optional[float], Optional[float], bool]:
    parser: ArgumentParser = ArgumentParser(description="Fine tuning runner")
    parser.add_argument("--gamma", nargs="+", type=float, help="Gamma value")
    parser.add_argument("--alpha", nargs="+", type=float, help="Alpha value")
    parser.add_argument("--lr", nargs="+", type=float, help="Learning rate value")
    parser.add_argument(
        "--store-checkpoints-in-temp-storage",
        action="store_true",
        help="If provided, checkpoints with saved models will be saved in temp storage",
    )

    try:
        args = parser.parse_args()
        return (
            args.gamma[0],
            args.alpha[0],
            args.lr[0],
            args.store_checkpoints_in_temp_storage,
        )
    except Exception:
        return None, None, None, False


if __name__ == "__main__":
    example_train(*parse_args())
