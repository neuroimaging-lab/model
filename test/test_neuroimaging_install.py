from __future__ import annotations

import importlib
from pathlib import Path

import nibabel as nib
from nibabel.nifti1 import Nifti1Image
import torch

DATA_ROOT = Path(__file__).resolve().parent.parent
IMAGE_PATH = (
    DATA_ROOT / "datasets" / "Task01_BrainTumour" / "imagesTr" / "BRATS_001.nii.gz"
)
LABEL_PATH = (
    DATA_ROOT / "datasets" / "Task01_BrainTumour" / "labelsTr" / "BRATS_001.nii.gz"
)
OUTPUT_DIR = DATA_ROOT / "test_outputs"

BEST_WEIGHTS_PATH = (
    DATA_ROOT / "datasets" / "Task01_BrainTumour" / "imagesTr" / "best_model.pth"
)


def test_public_api_exports() -> None:
    neuro = importlib.import_module("neuroimaging")

    expected_symbols = [
        "load_trained_UNet3D_model",
        "UNet3DModel",
        "UNet3DSegmenter",
        "UNet3DTransforms",
    ]

    missing = [symbol for symbol in expected_symbols if not hasattr(neuro, symbol)]
    assert not missing, f"Missing symbols on installed package: {', '.join(missing)}"


def test_prepare_input_pipeline(
    image_path: Path, label_path: Path, output_dir: Path
) -> torch.Tensor:
    neuro = importlib.import_module("neuroimaging")

    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))
    if not isinstance(image_nii, Nifti1Image):
        msg = f"Expected NIfTI input image, got {type(image_nii).__name__}"
        raise TypeError(msg)
    if not isinstance(label_nii, Nifti1Image):
        msg = f"Expected NIfTI label image, got {type(label_nii).__name__}"
        raise TypeError(msg)
    image = torch.from_numpy(image_nii.get_fdata()).float()
    label = torch.from_numpy(label_nii.get_fdata()).float()

    img = image.permute(3, 2, 0, 1).contiguous()  # (C, D, H, W)
    lbl = label.permute(2, 0, 1).unsqueeze(0).long().contiguous()  # (1, D, H, W)

    tfm = neuro.UNet3DTransforms.validation()
    out = tfm({"image": img, "label": lbl})
    if isinstance(out, list):
        out = out[0]

    t_img = out["image"].cpu()
    t_lbl = out["label"].cpu()

    output_dir.mkdir(parents=True, exist_ok=True)
    img_out = output_dir / "neuroimaging_transform_image.nii.gz"
    lbl_out = output_dir / "neuroimaging_transform_label.nii.gz"

    result_image = t_img.permute(2, 3, 1, 0).numpy()
    nib.save(nib.Nifti1Image(result_image, image_nii.affine), str(img_out))

    result_label = t_lbl.squeeze(0).permute(1, 2, 0).to(torch.int16).numpy()
    nib.save(nib.Nifti1Image(result_label, label_nii.affine), str(lbl_out))

    tensor = neuro.UNet3DSegmenter.prepare_input(img_out, device="cpu")

    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 4
    return tensor


def test_model_forward_pass(tensor: torch.Tensor, output_dir: Path) -> None:
    neuro = importlib.import_module("neuroimaging")
    output_dir.mkdir(parents=True, exist_ok=True)
    segmenter_weights = neuro.UNet3DSegmenter.from_state_dict(
        weights_path=BEST_WEIGHTS_PATH,
        device="cpu",
    )
    mask_path_weights = output_dir / "neuroimaging_smoke_mask_weights.nii.gz"
    mask = segmenter_weights.predict(tensor, output_path=mask_path_weights)
    segmenter = neuro.UNet3DSegmenter.from_packaged_checkpoint(device="cpu")
    mask_path = output_dir / "neuroimaging_smoke_mask.nii.gz"
    mask = segmenter.predict(tensor, output_path=mask_path)
    assert mask.shape == tuple(tensor.shape[-3:])
    assert mask_path.exists()


if __name__ == "__main__":
    test_public_api_exports()
    prepared = test_prepare_input_pipeline(IMAGE_PATH, LABEL_PATH, OUTPUT_DIR)
    test_model_forward_pass(prepared, OUTPUT_DIR)
    print("Neuroimaging library test passed.")
