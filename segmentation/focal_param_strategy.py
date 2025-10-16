import math
from typing import Dict, Literal

from segmentation.config import DATA_LABELS
from util.metric_saver import MetricSaver


class FocalParamStrategy:
    """
    Contains alpha, gamma and epsilon parameters for focal loss.
    Determines the alpha parameter based on class distribution and chosen strategy.
    """

    def __init__(
        self,
        class_proportions: Dict[str, float],
        metric_saver: MetricSaver,
        strategy: Literal["inverse", "inverse_sqrt", "log_scaling"] = "inverse",
        gamma: float = 2.0,
        eps: float = 1e-6,
    ):
        """Calculates alpha values based on the provided strategy and class proportions.
        Saves the parameters to a text file.
        """
        self._class_proportions: Dict[str, float] = class_proportions
        self.gamma: float = gamma
        self.eps: float = eps
        self.alpha: list[float] = self._compute_alpha(strategy)
        metric_saver.save_txt_file(
            f"strategy: {strategy}\ngamma: {self.gamma}\nalpha: {self.alpha}\n",
            filename="focal_loss_params.txt",
        )

    def _compute_alpha(
        self, strategy: Literal["inverse", "inverse_sqrt", "log_scaling"]
    ) -> list[float]:
        """
        Computes the alpha parameter for each class based on its proportion.
        """
        if strategy == "inverse":
            res_dict = self._calculate_inverse_frequency()
        elif strategy == "inverse_sqrt":
            res_dict = self._calculate_sqrt_inverse_frequency()
        elif strategy == "log_scaling":
            res_dict = self._calculate_log_scaling()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        print(f"Computed alpha values using '{strategy}' strategy:")
        result = []
        for label in DATA_LABELS:
            result.append(res_dict[label])
            print(f"{label}: {res_dict[label]:.4f}")

        return result

    def _calculate_inverse_frequency(self) -> dict[str, float]:
        inverse_freq = {
            label: 1.0 / prop for label, prop in self._class_proportions.items()
        }
        total = sum(inverse_freq.values())
        return {
            key: inverse_freq[key] / total for key in self._class_proportions.keys()
        }

    def _calculate_sqrt_inverse_frequency(self) -> dict[str, float]:
        sqrt_inverse_freq = {
            label: 1.0 / (prop**0.5) for label, prop in self._class_proportions.items()
        }
        total = sum(sqrt_inverse_freq.values())
        return {
            key: sqrt_inverse_freq[key] / total
            for key in self._class_proportions.keys()
        }

    def _calculate_log_scaling(self) -> dict[str, float]:
        log_scaled = {
            label: 1.0 / math.log(1.02 + prop)
            for label, prop in self._class_proportions.items()
        }
        total = sum(log_scaled.values())
        return {key: log_scaled[key] / total for key in self._class_proportions.keys()}
