from pathlib import Path
from torchsummary import summary  # type: ignore
import numpy as np
import torch
from torch.nn import functional as F

from segmentation.dataset import BrainTumorDataset
from segmentation.models.unet3d import UNet3D
from segmentation.transforms import train_transforms


def load_model(checkpoint_path: Path, device: str = "cpu") -> torch.nn.Module:
    model = UNet3D(
        in_channels=4,
        num_classes=4,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # THIS IS RESPONSIBLE FOR SCRIPTING THE MODEL, SAVING IT, AND LOADING IT BACK
    scripted_model = torch.jit.script(model)
    scripted_model.save(checkpoint_path.parent / "best_model_scripted.pt")
    loaded_model = torch.jit.load(checkpoint_path.parent / "best_model_scripted.pt")
    # END

    model.to(device)
    model.eval()

    print(
        f"Loaded model from epoch {checkpoint['epoch']} with validation Dice score: {checkpoint['val_dice']:.4f}"
    )

    return model


def load_single_sample(file_path: str, modalities=["FLAIR", "T1w", "t1gd", "T2w"]):
    dataset = BrainTumorDataset(
        root_dir=str(Path(file_path).parent.parent),
        split="test",
        modalities=modalities,
        transform=train_transforms,
        target_transform=train_transforms,
        cache_data=False,
    )

    image, _ = dataset[0]

    image = image.unsqueeze(0)

    return image


def predict(
    model: torch.nn.Module, scan: torch.Tensor, device: str = "cpu"
) -> np.ndarray:
    scan = scan.to(device)
    prediction: np.ndarray

    with torch.no_grad():
        output = model(scan)
        output = F.softmax(output, dim=1)

        prediction_tens = torch.argmax(output, dim=1)

    prediction = prediction_tens.cpu().numpy()[0]

    return prediction


def example_predict(model_dir: str) -> np.ndarray:
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    dataset_dir = project_root / "datasets" / "Task01_BrainTumour"

    sample_image = dataset_dir / "imagesTr" / "BRATS_001.nii.gz"

    try:
        scan = load_single_sample(str(sample_image))
        print(f"Loaded scan with shape: {scan.shape}")
    except Exception as e:
        print(f"Error loading scan: {e}")
        return np.array([])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(
        checkpoint_path=project_root / "checkpoints" / model_dir / "best_model.pth",
        device=device,
    )

    prediction = predict(model, scan, device)
    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique segmentation classes: {np.unique(prediction)}")

    return prediction


if __name__ == "__main__":
    model_dir = "run_20251011_114333"
    example_predict(model_dir=model_dir)
