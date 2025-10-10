import os

from segmentation.config import METRICS_DIR


class MetricSaver:
    def __init__(self):
        os.makedirs(METRICS_DIR, exist_ok=True)
        self.file_path = os.path.join(METRICS_DIR, "metrics.csv")
        self.header_name = "Epoch,Train Dice,Val Dice\n"
        with open(self.file_path, mode="a", encoding="utf-8") as f:
            f.write(self.header_name)

    def save(self, epoch: int, train_dice: float, val_dice: float):
        entry = (
            f"{epoch:03d},{train_dice:.4f},{val_dice:.4f}\n"
        )
        with open(self.file_path, mode="a", encoding="utf-8") as f:
            f.write(entry)
