import os

from segmentation.config import METRICS_DIR


class MetricSaver:
    def __init__(self):
        os.makedirs(METRICS_DIR, exist_ok=True)
        self.file_path = os.path.join(METRICS_DIR, "metrics.csv")
        self.header_name = (
            "Epoch;Train dice;Val dice;Train per class dice;Val per class dice\n"
        )
        with open(self.file_path, mode="a", encoding="utf-8") as f:
            f.write(self.header_name)

    def save_entry(
        self,
        epoch: int,
        train_dice: float,
        train_dice_per_class: list[float],
        val_dice: float,
        val_dice_per_class: list[float],
    ):
        train_dice_per_class_str = (
            "[" + ", ".join(f"{x:.4f}" for x in train_dice_per_class) + "]"
        )
        val_dice_per_class_str = (
            "[" + ", ".join(f"{x:.4f}" for x in val_dice_per_class) + "]"
        )

        entry = f"{epoch:03d};{train_dice:.4f};{val_dice:.4f};{train_dice_per_class_str};{val_dice_per_class_str}\n"
        with open(self.file_path, mode="a", encoding="utf-8") as f:
            f.write(entry)

    def save_txt_file(self, text: str, filename: str = "class_distribution.txt"):
        txt_path = os.path.join(METRICS_DIR, filename)
        with open(txt_path, mode="w", encoding="utf-8") as f:
            f.write(text)
