from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from typing import Iterable, Optional, Sequence

from monai.transforms import (
    CastToTyped,
    Compose,
    Compose as ComposeType,
    DivisiblePadD,
    NormalizeIntensityd,
)
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import numpy as np
import torch
from torch import nn

from library.neuroimaging.model_loader import (
    load_trained_UNet3D_model,
)
from segmentation.models.unet3d import UNet3D


class UNet3DTransforms:
    """Factory for MONAI Compose pipelines aligned with the training/validation recipes."""

    @classmethod
    def normalize_input(cls) -> "ComposeType":
        return Compose(
            [
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                DivisiblePadD(keys=["image"], k=16),
            ]
        )

    @classmethod
    def validation(cls) -> "ComposeType":
        return Compose(
            [
                NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
                CastToTyped(keys=["label"], dtype=torch.int64),
                DivisiblePadD(keys=["image", "label"], k=16),
            ]
        )


class UNet3DModel(UNet3D):
    """Thin wrapper around the raw UNet3D architecture with convenience constructors."""

    DEFAULT_LEVEL_CHANNELS: Sequence[int] = (64, 128, 256)
    DEFAULT_BOTTLENECK_CHANNELS: int = 512

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        level_channels: Optional[Iterable[int]] = None,
        bottleneck_channels: Optional[int] = None,
        dropout_rate: float = 0.0,
    ) -> None:
        resolved_level_channels = list(level_channels or self.DEFAULT_LEVEL_CHANNELS)
        resolved_bottleneck = (
            bottleneck_channels
            if bottleneck_channels is not None
            else self.DEFAULT_BOTTLENECK_CHANNELS
        )
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            level_channels=resolved_level_channels,
            bottleneck_channels=resolved_bottleneck,
            dropout_rate=dropout_rate,
        )

    @classmethod
    def from_weights(
        cls,
        weights_path: Path,
        device: str = "cpu",
        map_location: str = "cpu",
        strict: bool = True,
        **kwargs,
    ) -> "UNet3DModel":
        """Load a state-dict based checkpoint into a freshly initialised network."""
        model = cls(**kwargs)
        checkpoint = torch.load(Path(weights_path), map_location=map_location)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        new_dict = {}
        for key, value in state_dict.items():
            cleaned_key = (
                key.replace("module.", "") if key.startswith("module.") else key
            )
            new_dict[cleaned_key] = value

        model.load_state_dict(new_dict, strict=strict)
        model.to(device)
        model.eval()
        return model


class UNet3DSegmenter:
    """High-level inference helper mirroring the ergonomics of other ML libraries."""

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_packaged_checkpoint(cls, device: str = "cpu") -> "UNet3DSegmenter":
        """Load the TorchScript model."""
        model = load_trained_UNet3D_model(device=device)
        return cls(model=model, device=device)

    @classmethod
    def from_state_dict(
        cls,
        weights_path: Path,
        device: str = "cpu",
        **model_kwargs,
    ) -> "UNet3DSegmenter":
        """Instantiate a native PyTorch UNet3D model from weights."""
        model = UNet3DModel.from_weights(
            weights_path=weights_path, device=device, **model_kwargs
        )
        return cls(model=model, device=device)

    def predict(
        self,
        x: torch.Tensor,
        output_path: Optional[Path] = Path("predicted_mask.nii.gz"),
    ):
        with torch.no_grad():
            output = self.model(x.to(self.device))

        mask = output.argmax(dim=1).squeeze().cpu().numpy()

        if output_path is not None:
            mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine=np.eye(4))
            nib.save(mask_img, output_path)

        return mask

    @staticmethod
    def prepare_input(image_path: Path, device: str = "cpu"):
        img = nib.load(str(image_path))
        if not isinstance(img, Nifti1Image):
            msg = f"Expected NIfTI input image, got {type(img).__name__}"
            raise TypeError(msg)
        data = img.get_fdata()

        if data.ndim == 3:
            data = np.expand_dims(data, axis=-1)

        data = np.transpose(data, (3, 2, 0, 1))
        x = torch.from_numpy(data).unsqueeze(0).float().to(device)  # (1, C, D, H, W)

        return x
