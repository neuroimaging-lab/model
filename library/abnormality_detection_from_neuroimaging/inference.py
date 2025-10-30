# ruff: noqa: E402
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

import nibabel as nib
import numpy as np
import torch

from library.abnormality_detection_from_neuroimaging.model_loader import (
    load_trained_3D_U_Net_model,
)
from segmentation.models.unet3d import UNet3D


def predict_3D_U_Net(
    x: torch.Tensor, output_path: Path = Path("predicted_mask.nii.gz"), device="cpu"
):
    model = load_trained_3D_U_Net_model(device=device)
    model.eval()
    print("[INFO] Model loaded")

    print("[INFO] Processing...")
    with torch.no_grad():
        output = model(x)

    mask = output.argmax(dim=1).squeeze().cpu().numpy()

    mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine=np.eye(4))
    nib.save(mask_img, output_path)
    print("[INFO] Predicted mask saved to:", output_path)

    return mask


def model_3D_U_Net(
    in_channels=4, num_classes=4, level_channels=[64, 128, 256], bottleneck_channels=512
):
    return UNet3D(
        in_channels=in_channels,
        num_classes=num_classes,
        level_channels=level_channels,
        bottleneck_channels=bottleneck_channels,
    )


def model_3D_U_Net_weights(
    weights_path: Path,
    in_channels=4,
    num_classes=4,
    level_channels=[64, 128, 256],
    bottleneck_channels=512,
):
    model = model_3D_U_Net(
        in_channels, num_classes, level_channels, bottleneck_channels
    )
    dict = torch.load(weights_path, map_location="cpu")["model_state_dict"]

    new_dict = {}
    for k, v in dict.items():
        key = k.replace("module.", "") if k.startswith("module.") else k
        new_dict[key] = v

    model.load_state_dict(new_dict)
    model.eval()
    return model


def prepare_input_3D_U_Net(image_path: Path, device="cpu", multiple=8):
    img = nib.load(str(image_path))
    data = img.get_fdata()
    print(f"[INFO] Raw data shape: {data.shape}")

    if data.ndim == 3:
        data = np.expand_dims(data, axis=-1)

    data = np.transpose(data, (3, 2, 0, 1))
    x = torch.from_numpy(data).unsqueeze(0).float().to(device)  # (1, C, D, H, W)
    x = pad_to_multiple_of(x, multiple=multiple)
    print(f"[INFO] Preprocessed tensor shape: {x.shape}")

    return x


def pad_to_multiple_of(x: torch.Tensor, multiple=8):
    D, H, W = x.shape[-3:]
    pad_d = (multiple - D % multiple) % multiple
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    padding = (0, pad_w, 0, pad_h, 0, pad_d)
    return torch.nn.functional.pad(x, padding)
