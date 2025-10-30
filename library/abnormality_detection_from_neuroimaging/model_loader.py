from pathlib import Path

import torch


def load_trained_3D_U_Net_model(model_filename="3DU-Net-model.pt", device="cpu"):
    package_dir = Path(__file__).parent
    local_path = package_dir / model_filename

    print(f"[INFO] Using model: {local_path}")
    model_path = local_path

    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model
