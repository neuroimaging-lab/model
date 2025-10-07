import os
import random
from typing import Optional

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch

from segmentation.config import METRICS_DIR
from matplotlib.colors import ListedColormap


class GraphMaker:
    def __init__(self):
        self.fig = None
        self.pred_slice = None
        self.target_slice = None

        self.slices_dir_path = f"{METRICS_DIR}slices/"
        self.slice_idx = None
        os.makedirs(self.slices_dir_path)

    def _find_slice_idx_with_all_classes(self, target_labels: torch.Tensor) -> Optional[int]:
        random.seed(42)

        target_labels = target_labels.squeeze(1)  # [B, D, H, W]
        num_slices = target_labels.shape[1]  # Number of slices in depth dimension

        # Generate and shuffle indices
        indices = list(range(num_slices))
        random.shuffle(indices)

        for desired_class in range(4, 2, -1):
            for slice_idx in indices:
                slice_data = target_labels[0, slice_idx, :, :].cpu().detach().numpy()
                unique_classes = np.unique(slice_data)
                if len(unique_classes) == desired_class:
                    return slice_idx

        raise ValueError(
                 f"Wrong target_labels: No slice contains all classes in {num_slices} slices."
             )

    def set_prediction_and_ground_truth_slice(
        self,
        pred_logits: torch.Tensor,
        target_labels: torch.Tensor,
        slice_idx: Optional[int] = None,
    ):
        pred_labels = torch.argmax(pred_logits, dim=1)  # [B, D, H, W]
        target_labels = target_labels.squeeze(1)  # [B, D, H, W]

        # slice_idx = (
        #     #pred_labels.shape[1] // 2 if slice_idx is None else slice_idx
        #     pred_labels.shape[1] - 1 if slice_idx is None else slice_idx
        # )  # middle one as a default
        # if slice_idx < 0 or slice_idx >= pred_labels.shape[1]:
        #     raise ValueError(
        #         f"Slice_idx {slice_idx} is out of bounds for depth dimension {pred_labels.shape[1]}"
        #     )
        if slice_idx is None:
            self.slice_idx = self._find_slice_idx_with_all_classes(target_labels) if slice_idx is None else slice_idx
            print(f"\nChosen slice index {self.slice_idx}")

        self.pred_slice = pred_labels[0, self.slice_idx, :, :].cpu().detach().numpy()
        self.target_slice = target_labels[0, self.slice_idx, :, :].cpu().detach().numpy()

    def visualize_slice(self, epoch: int, dice_score: float):
        self._create_slice_figure(epoch, dice_score)
        plt.show()

    def save_slice(self, epoch: int, dice_score: float):
        slice_path = self.slices_dir_path + f"slice_{epoch}.png"
        fig = self._create_slice_figure(epoch, dice_score)
        fig.savefig(slice_path)
        plt.close(fig)

    def _create_slice_figure(self, epoch: int, dice_score: float):

        colors = ["darkviolet", "blue", "green", "yellow"] 
        labels = ["Background", "Edema", "Non-enhancing tumor", "Enhancing tumour"]

        ticks = [0, 1, 2, 3]
        cmap = ListedColormap(colors)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        title = f"Epoch {epoch} — Dice Score: {dice_score:.4f}"
        fig.suptitle(title, fontsize=16)

        # Prediction slice
        axes[0].set_title("Prediction")
        im0 = axes[0].imshow(self.pred_slice, cmap=cmap, vmin=0, vmax=3)
        cbar0 = fig.colorbar(im0, ax=axes[0], ticks=ticks)
        cbar0.ax.set_yticklabels(labels)

        # Ground truth slice
        axes[1].set_title("Ground Truth")
        im1 = axes[1].imshow(self.target_slice, cmap=cmap, vmin=0, vmax=3)
        cbar1 = fig.colorbar(im1, ax=axes[1], ticks=ticks)
        cbar1.ax.set_yticklabels(labels)

        fig.tight_layout()
        return fig

    def make_summary_of_slices(
        self,
        epochs: int,
        step: int = 20,
        grid_cols: int = 2,
        output_file: str = "summary.png",
    ):
        slice_indices = list(range(0, epochs + 1, step))
        image_paths = [
            os.path.join(self.slices_dir_path, f"slice_{i}.png") for i in slice_indices
        ]

        images = [Image.open(path) for path in image_paths if os.path.exists(path)]

        if not images:
            print("No images to combine.")
            return

        img_width, img_height = images[0].size
        grid_rows = (len(images) + grid_cols - 1) // grid_cols
        summary_img = Image.new(
            "RGB", (grid_cols * img_width, grid_rows * img_height), color="white"
        )

        for idx, img in enumerate(images):
            row = idx // grid_cols
            col = idx % grid_cols
            x = col * img_width
            y = row * img_height
            summary_img.paste(img, (x, y))

        summary_img.save(os.path.join(METRICS_DIR, output_file))
