import ast
import csv
import os
import random
from typing import Literal, Optional

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from segmentation.config import DATA_LABELS, METRICS_DIR


class GraphMaker:
    def __init__(self):
        self.fig = None
        self.pred_slice = None
        self.target_slice = None

        self.slices_dir_path = f"{METRICS_DIR}slices/"
        self.slices_train_path = self.slices_dir_path + "train/"
        self.slices_val_path = self.slices_dir_path + "val/"

        self.slice_idx = None
        self.metrics_path: str = METRICS_DIR + "metrics.csv"

        max_distance_in_mri = (160**2 + 240**2 + 240**2) ** 0.5
        self.INF_PLOT_REPRESENTATION = int(max_distance_in_mri)

        os.makedirs(self.slices_dir_path)
        os.makedirs(self.slices_train_path)
        os.makedirs(self.slices_val_path)

    def _find_slice_idx_with_all_classes(
        self, target_labels: torch.Tensor
    ) -> Optional[int]:
        random.seed(1)

        target_labels = target_labels.squeeze(1)  # [B, D, H, W]
        num_slices = target_labels.shape[1]

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

        if slice_idx is None:
            self.slice_idx = (
                self._find_slice_idx_with_all_classes(target_labels)
                if slice_idx is None
                else slice_idx
            )
            print(f"\nChosen slice index {self.slice_idx}")

        self.pred_slice = pred_labels[0, self.slice_idx, :, :].cpu().detach().numpy()
        self.target_slice = (
            target_labels[0, self.slice_idx, :, :].cpu().detach().numpy()
        )

    def save_slice(
        self,
        epoch: int,
        dice_score: float,
        to: Literal["train", "val"],
        save_every_which_epoch: int = 5,
    ):
        if epoch % save_every_which_epoch == 0:
            slice_folder_path = (
                self.slices_train_path if to == "train" else self.slices_val_path
            )
            slice_path = f"{slice_folder_path}slice_{epoch}.png"
            fig = self._create_slice_figure(epoch, dice_score)
            fig.savefig(slice_path)
            plt.close(fig)

    def _create_slice_figure(self, epoch: int, dice_score: float):
        colors = ["darkviolet", "blue", "green", "yellow"]

        ticks = [0, 1, 2, 3]
        cmap = ListedColormap(colors)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        title = f"Epoch {epoch} — Dice Score: {dice_score:.4f}"
        fig.suptitle(title, fontsize=16)

        # Prediction slice
        axes[0].set_title("Prediction")
        im0 = axes[0].imshow(self.pred_slice, cmap=cmap, vmin=0, vmax=3)
        cbar0 = fig.colorbar(im0, ax=axes[0], ticks=ticks)
        cbar0.ax.set_yticklabels(DATA_LABELS)

        # Ground truth slice
        axes[1].set_title("Ground Truth")
        im1 = axes[1].imshow(self.target_slice, cmap=cmap, vmin=0, vmax=3)
        cbar1 = fig.colorbar(im1, ax=axes[1], ticks=ticks)
        cbar1.ax.set_yticklabels(DATA_LABELS)

        fig.tight_layout()
        return fig

    def make_summary_of_results(
        self,
        epochs: int,
        dice_plot_file: str = "dice_plot.png",
        dice_per_class_file: str = "dice_per_class_plot.png",
        hausdorff_plot_file: str = "hausdorff_plot.png",
        hausdorff_per_class_file: str = "hausdorff_per_class_plot.png",
    ):
        plot_dice = self._make_summary_plot("Dice")
        plot_dice.savefig(os.path.join(METRICS_DIR, dice_plot_file))

        plot_dice_per_class = self._make_per_class_summary_plot("Dice")
        plot_dice_per_class.savefig(os.path.join(METRICS_DIR, dice_per_class_file))

        summary_train_img = self._prepare_slice_summary_image(epochs, to="train")
        summary_train_img.save(os.path.join(METRICS_DIR, "summary_train.png"))

        summary_val_img = self._prepare_slice_summary_image(epochs, to="val")
        summary_val_img.save(os.path.join(METRICS_DIR, "summary_val.png"))

        plot_hausdorff = self._make_summary_plot("Hausdorff")
        plot_hausdorff.savefig(os.path.join(METRICS_DIR, hausdorff_plot_file))

        plot_hausdorff_per_class = self._make_per_class_summary_plot("Hausdorff")
        plot_hausdorff_per_class.savefig(
            os.path.join(METRICS_DIR, hausdorff_per_class_file)
        )

    def _prepare_slice_summary_image(
        self, epochs, to: Literal["train", "val"], step: int = 20, grid_cols: int = 2
    ):
        dir_path = self.slices_train_path if to == "train" else self.slices_val_path
        image_paths = [
            os.path.join(dir_path, f"slice_{i}.png") for i in range(0, epochs + 1, step)
        ]

        images = [Image.open(path) for path in image_paths if os.path.exists(path)]

        if not images:
            raise ValueError(
                "Could not prepare summary image. No images found for the specified epochs."
            )

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

        return summary_img

    def _make_summary_plot(self, metric: Literal["Dice", "Hausdorff"]):
        data = np.genfromtxt(self.metrics_path, delimiter=";", skip_header=1)

        epochs = data[:, 0]
        if metric == "Dice":
            train_avg = data[:, 1]
            val_avg = data[:, 2]
        elif metric == "Hausdorff":
            train_avg = data[:, 5]
            train_avg = np.where(
                np.isinf(train_avg), self.INF_PLOT_REPRESENTATION, train_avg
            )
            val_avg = data[:, 6]
            val_avg = np.where(np.isinf(val_avg), self.INF_PLOT_REPRESENTATION, val_avg)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            epochs,
            train_avg,
            label=f"Train {metric}",
            color="blue",
            linewidth=2,
            marker="o",
        )
        ax.plot(
            epochs,
            val_avg,
            label=f"Validation {metric}",
            color="orange",
            linewidth=2,
            marker="s",
        )

        ax.set_title(f"{metric} Over Epochs", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(
            f"{metric} {'Score' if metric == 'Dice' else 'Distance'}", fontsize=12
        )
        if metric == "Dice":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(0, self.INF_PLOT_REPRESENTATION)

        min_epoch, max_epoch, step = self._get_epochs_info(epochs)
        ax.set_xlim(min_epoch, max_epoch + 0.5)
        ax.set_xticks(np.arange(min_epoch, max_epoch + 1, step))
        ax.tick_params(axis="x", rotation=45)

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=10, loc="lower right")

        plt.tight_layout()

        return fig

    def _make_per_class_summary_plot(self, metric: Literal["Dice", "Hausdorff"]):
        epochs = []
        train_per_class = []
        val_per_class = []

        with open(self.metrics_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            next(reader)
            for data_row in reader:
                epochs.append(int(data_row[0]))
                if metric == "Dice":
                    train_per_class.append(ast.literal_eval(data_row[3]))
                    val_per_class.append(ast.literal_eval(data_row[4]))
                elif metric == "Hausdorff":
                    train_vals = ast.literal_eval(
                        data_row[7].replace("inf", str(self.INF_PLOT_REPRESENTATION))
                    )
                    val_vals = ast.literal_eval(
                        data_row[8].replace("inf", str(self.INF_PLOT_REPRESENTATION))
                    )
                    train_per_class.append(train_vals)
                    val_per_class.append(val_vals)

        epochs_arr = np.array(epochs)
        train_per_class_arr = np.array(train_per_class)
        val_per_class_arr = np.array(val_per_class)

        if metric == "Dice":
            num_classes = len(DATA_LABELS)
            row, col = (2, 2)
        elif metric == "Hausdorff":
            num_classes = len(DATA_LABELS) - 1
            row, col = (num_classes, 1)

        fig, axes = plt.subplots(row, col, figsize=(10, 8 if metric == "Dice" else 12))
        axes = axes.flatten()

        for i in range(num_classes):
            ax = axes[i]

            ax.plot(
                epochs_arr,
                train_per_class_arr[:, i],
                label=f"Train {metric}",
                color="blue",
                linewidth=2,
                marker="o",
            )
            ax.plot(
                epochs_arr,
                val_per_class_arr[:, i],
                label=f"Validation {metric}",
                color="orange",
                linewidth=2,
                marker="s",
            )
            class_idx = i if metric == "Dice" else i + 1
            ax.set_title(f"{DATA_LABELS[class_idx]}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel(
                f"{metric} {'Score' if metric == 'Dice' else 'Distance'}", fontsize=10
            )
            if metric == "Dice":
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(0, self.INF_PLOT_REPRESENTATION)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(fontsize=9, loc="lower right")

        plt.suptitle(
            f"Per-Class {metric} {'Score' if metric == 'Dice' else 'Distance'} During Training and Validation",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        return fig

    def _get_epochs_info(self, epochs):
        min_epoch = int(np.min(epochs))
        max_epoch = int(np.max(epochs))
        num_epochs = max_epoch - min_epoch + 1
        if num_epochs <= 20:
            step = 1
        elif num_epochs <= 50:
            step = 2
        elif num_epochs <= 100:
            step = 5
        elif num_epochs <= 200:
            step = 10
        elif num_epochs <= 500:
            step = 20
        else:
            step = 50

        return min_epoch, max_epoch, step
