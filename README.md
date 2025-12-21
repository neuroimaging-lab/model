# 3D U-Net - Brain Tumor Segmentation

This repository provides an implementation of the 3D U-Net model architecture, based on the Google DeepMind paper [arxiv](https://arxiv.org/abs/1606.06650). The model was trained using [medicaldecathlon dataset](http://medicaldecathlon.com/) on the AGH Cyfronet cluster using an NVIDIA A100-SXM4-40GB accelerator, with 500 epochs taking approximately 28 hours to complete with 480 MRI scans achieving 0.8335 Dice Score.

## Library

The trained model is available via the [PyPI library](https://pypi.org/project/neuroimaging/) for easy loading and inference on your own data. The library also includes the model architecture, allowing you to experiment with training and fine-tuning. 

Example code snippet:

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
