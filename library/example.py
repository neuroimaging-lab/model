from pathlib import Path

from abnormality_detection_from_neuroimaging import (
    load_trained_3D_U_Net_model,
    model_3D_U_Net,
    model_3D_U_Net_weights,
    predict_3D_U_Net,
    prepare_input_3D_U_Net,
)
import matplotlib.pyplot as plt
import torch

image_path = Path("library/BRATS_001.nii.gz")
device = "cuda" if torch.cuda.is_available() else "cpu"

x = prepare_input_3D_U_Net(image_path, device=device)
model_3D_U_Net_trained = load_trained_3D_U_Net_model(device=device)
model = model_3D_U_Net()
model_weights = model_3D_U_Net_weights(
    weights_path=Path("library/best_model_weights.pth")
)

with torch.no_grad():
    mask = predict_3D_U_Net(x, output_path=Path("predicted_mask.nii.gz"), device=device)

with torch.no_grad():
    out_trained = model_3D_U_Net_trained(x)

with torch.no_grad():
    model_weights.to(device)
    out_weighted = model_weights(x)

slice_idx = mask.shape[0] // 2
plt.imshow(mask[slice_idx, :, :].T)
plt.show()
