import numpy as np
from torch.utils.data import DataLoader

from segmentation.config import DATA_LABELS


class ClassDistributionAnalyzer:
    def __init__(self, train_dataloader: DataLoader, validation_dataloader: DataLoader):
        self.proportions: dict = {}
        self.total_voxels: int = 1

        print("\nCalculating class distribution in provided dataset")
        self._analyze(train_dataloader, validation_dataloader)

    def print_probability_of_each_class(self):
        print("\nClass proportions in the provided dataset:")
        for label, proportion in self.proportions.items():
            print(f"'{label}': {proportion:.7f}")
        print()

    def _analyze(self, train_dataloader: DataLoader, validation_dataloader: DataLoader):
        print("Analyzing training dataset...")
        train_summ_dict, train_iters = self._get_sum_of_each_class_dict(
            train_dataloader
        )
        print("Analyzing validation dataset...")
        val_summ_dict, val_iters = self._get_sum_of_each_class_dict(
            validation_dataloader
        )

        self.proportions = {
            label: (train_summ_dict.get(label, 0) + val_summ_dict.get(label, 0))
            / ((train_iters + val_iters) * self.total_voxels)
            for label in DATA_LABELS
        }

    def _get_sum_of_each_class_dict(self, dataloader: DataLoader) -> tuple[dict, int]:
        summ_dict: dict = {}
        iterations: int = 0
        for _, masks in dataloader:
            for mask in masks:
                mask_np: np.ndarray = mask.squeeze(0).cpu().numpy()
                unique_classes, counts = np.unique(mask_np, return_counts=True)

                self.total_voxels = mask_np.size
                iterations += 1

                for class_idx, count in zip(unique_classes, counts):
                    summ_dict[DATA_LABELS[class_idx]] = (
                        summ_dict.get(DATA_LABELS[class_idx], 0) + count
                    )

        return summ_dict, iterations
