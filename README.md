# Model

## Setup

### Install uv

https://docs.astral.sh/uv/getting-started/installation/


```bash
uv venv                                     # creates venv
uv sync --all-extras --dev                  # installs dependencies on venv
uv run -m segmentation.examples.models      # this is how to run .py files
```

Format code with ruff
```bash
uv run ruff check --select I --fix    # format imports, or run without --fix to check only
uv run ruff format                    # format code, or run with --check
```

Check typing
```bash
uv run mypy . --config-file pyproject.toml  # runs type linter
```

## Library quickstart

Minimal, copy-pasteable way to predict the packaged TorchScript model on your own NIfTI image

```python
from pathlib import Path

import torch

from neuroimaging import UNet3DSegmenter, UNet3DTransforms

image_path = Path("path/to/your_scan.nii.gz")
device = "cpu"

input_tensor = UNet3DSegmenter.prepare_input(image_path, device=device)

tfm = UNet3DTransforms.normalize_input()
sample = {"image": input_tensor.squeeze(0)}
normalized = tfm(sample)
input_tensor = normalized["image"].unsqueeze(0).to(device)

segmenter = UNet3DSegmenter.from_packaged_checkpoint(device=device)
mask = segmenter.predict(
    input_tensor, output_path=image_path.with_name("predicted_mask.nii.gz")
)
```
