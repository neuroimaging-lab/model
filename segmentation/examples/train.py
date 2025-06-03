import torch

from segmentation.models.unet3d import UNet3D
from segmentation.train import dice_loss, train


def example_train():
    model = UNet3D(
        in_channels=4,
        num_classes=4,
        level_channels=[16, 32, 64],
        bottleneck_channels=128,
    )

    criterion = dice_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train(model, criterion, optimizer)


if __name__ == "__main__":
    example_train()
