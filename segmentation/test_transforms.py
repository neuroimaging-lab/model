from pathlib import Path

import nibabel as nib
import torch

from segmentation.transforms import train_transforms, val_transforms


def load_nifti(path):
    nii = nib.load(str(path))
    data = nii.get_fdata()
    return torch.tensor(data, dtype=torch.float32), nii.affine


def save_nifti(data, affine, path):
    nib.save(nib.Nifti1Image(data.numpy(), affine), str(path))


def test_train_transforms(
    img_path, mask_path, tmp_path: Path = Path("test_outputs"), use_train: bool = False
):
    tmp_path.mkdir(exist_ok=True)

    image, affine = load_nifti(img_path)
    label, _ = load_nifti(mask_path)

    img = image.permute(3, 2, 0, 1).contiguous()  # (C, D, H, W)
    lbl = label.permute(2, 0, 1).unsqueeze(0).long().contiguous()  # (1, D, H, W)

    data = {"image": img, "label": lbl}

    print("Original image shape: ", img.shape)
    print("Original label shape:", lbl.shape)

    tfm = train_transforms if use_train else val_transforms
    out = tfm(data)

    if isinstance(out, list):
        out = out[0]

    t_img = out["image"].cpu()
    t_lbl = out["label"].cpu()

    print("Transformed image shape :", t_img.shape)
    print("Transformed label shape :", t_lbl.shape)

    # Save image as (H, W, D, C)
    result_image = t_img.permute(2, 3, 1, 0)
    save_nifti(result_image, affine, tmp_path / "aug_image.nii.gz")

    # Save label as (H, W, D)
    result_label = t_lbl.squeeze(0).permute(1, 2, 0).to(torch.int16)
    save_nifti(result_label, affine, tmp_path / "aug_label.nii.gz")


image_path = (
    Path(__file__).parent.parent
    / "datasets"
    / "Task01_BrainTumour"
    / "imagesTr"
    / "BRATS_001.nii.gz"
)
label_path = (
    Path(__file__).parent.parent
    / "datasets"
    / "Task01_BrainTumour"
    / "labelsTr"
    / "BRATS_001.nii.gz"
)

if __name__ == "__main__":
    test_train_transforms(image_path, label_path, Path("test_outputs"), use_train=False)
