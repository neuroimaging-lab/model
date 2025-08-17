import os

from segmentation.config import METRICS_DIR


class MetricSaver:
    def __init__(self):
        os.makedirs(METRICS_DIR, exist_ok=True)
        self.file_path = os.path.join(METRICS_DIR, "metrics.txt")

    def save(self, epoch: int, train_dice: float, val_dice: float):
        entry = (
            f"Epoch {epoch:03d}:\n"
            f"  Train Dice: {train_dice:.4f}\n"
            f"  Val Dice:   {val_dice:.4f}\n"
            f"{'-' * 40}\n"
        )
        with open(self.file_path, mode="a", encoding="utf-8") as f:
            f.write(entry)
