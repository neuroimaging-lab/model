# Library quickstart

Minimal, copy-pasteable way to predict the packaged TorchScript model on your own NIfTI image

```python
from pathlib import Path

from neuroimaging import VolumeSegmenter, Transforms

image_path = Path("path/to/your_scan.nii.gz")
device = "cpu"

input_tensor = VolumeSegmenter.prepare_input(image_path, device=device)

preprocessor = Transforms.normalize_input()
normalized_data = preprocessor({"image": input_tensor})
input_tensor = normalized_data["image"].to(device)

segmenter = VolumeSegmenter.from_pretrained(device=device)
mask = segmenter.predict(
    input_tensor, output_path=image_path.with_name("predicted_mask.nii.gz")
)
```

# Model

## Development

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
