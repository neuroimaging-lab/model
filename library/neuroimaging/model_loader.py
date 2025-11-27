from __future__ import annotations

from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download  # type: ignore[import]
import torch

HF_REPO_ID = "N1T1/neuroimaging_UNet3D"
HF_FILENAME = "UNet3D-model.pt"


def load_trained_UNet3D_model(
    model_filename: str = HF_FILENAME,
    *,
    device: str = "cpu",
    repo_id: str = HF_REPO_ID,
    revision: Optional[str] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    cache_dir: Optional[Path] = None,
    prefer_local: bool = True,
) -> torch.jit.ScriptModule:
    """
    Load the TorchScript UNet3D model, downloading it from Hugging Face if needed.
    """
    package_dir = Path(__file__).parent
    local_path = package_dir / model_filename

    if prefer_local and local_path.exists():
        model_path = local_path
    else:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=model_filename,
            revision=revision,
            force_download=force_download,
            local_files_only=local_files_only,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        model_path = Path(downloaded_path)

    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model
