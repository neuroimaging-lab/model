from datetime import datetime
import os
from typing import Optional

import matplotlib.pyplot as plt
import torch


class GraphMaker:
    def __init__(self):
        self.fig = None
        self.pred_slice = None
        self.target_slice = None

        timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.FILE_PATH = f"slices/{timestamp}/"

        os.makedirs(self.FILE_PATH)

    def set_prediction_and_ground_truth_slice(
        self,
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
        slice_idx: Optional[int] = None,
    ):
        pred_labels = torch.argmax(pred_logits, dim=1)  # [B, D, H, W]
        target_labels = target_labels.squeeze(1)  # [B, D, H, W]

        slice_idx = (
            pred_labels.shape[1] // 2 if slice_idx is None else slice_idx
        )  # middle one as a default
        if slice_idx < 0 or slice_idx >= pred_labels.shape[1]:
            raise ValueError(
                f"Slice_idx {slice_idx} is out of bounds for depth dimension {pred_labels.shape[1]}"
            )

        self.pred_slice = pred_labels[0, slice_idx, :, :].cpu().detach().numpy()
        self.target_slice = target_labels[0, slice_idx, :, :].cpu().detach().numpy()

    def visualize_slice(self, epoch: int, dice_score: float):
        self._create_slice_figure(epoch, dice_score)
        plt.show()

    def save_slice(self, epoch: int, dice_score: float):
        filepath = self.FILE_PATH + f"slice_{epoch}.png"
        fig = self._create_slice_figure(epoch, dice_score)
        fig.savefig(filepath)
        plt.close(fig)

    def _create_slice_figure(self, epoch: int, dice_score: float):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        title = f"Epoch {epoch} — Dice Score: {dice_score:.4f}"
        fig.suptitle(title, fontsize=16)

        axes[0].set_title("Prediction")
        im0 = axes[0].imshow(self.pred_slice, cmap="viridis")
        fig.colorbar(im0, ax=axes[0])

        axes[1].set_title("Ground Truth")
        im1 = axes[1].imshow(self.target_slice, cmap="viridis")
        fig.colorbar(im1, ax=axes[1])

        fig.tight_layout()
        return fig
