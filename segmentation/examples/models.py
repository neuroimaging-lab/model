from torchsummary import summary  # type: ignore

from segmentation.models.unet3d import UNet3D


def unet3d():
    model = UNet3D(
        in_channels=3,
        num_classes=1,
    )
    summary(model=model, input_size=(3, 16, 128, 128), batch_size=-1, device="cpu")


if __name__ == "__main__":
    unet3d()
