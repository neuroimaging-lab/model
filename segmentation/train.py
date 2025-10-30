from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation.config import (
    CLUSTER_TRAINING_ENABLED,
    CURR_RUN,
    SAVE_BEST_MODEL,
    get_temp_storage_path,
)
from segmentation.focal_param_strategy import FocalParamStrategy
from util.class_distribution_analyzer import ClassDistributionAnalyzer
from util.graph_maker import GraphMaker
from util.metric_saver import MetricSaver
from util.train_support import (
    create_dataloaders,
    prepare_images_and_masks,
    save_model,
)


class Trainer:
    """
    Trainer class for training and validating a brain tumor segmentation model.
    Trains with multi-class Dice loss, and reports mean Dice (optionally excluding background).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        batch_size: int = 1,
        val_split: float = 0.2,
        save_dir: Path | str = "checkpoints",
        num_workers: int = 2,
        pin_memory: bool = True,
        store_checkpoints_in_temp_storage: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer

        self.data_config: Dict[str, Any] = {
            "batch_size": batch_size,
            "val_split": val_split,
            "save_dir": Path(save_dir),
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }

        self.graph_maker = GraphMaker()
        self.metric_saver = MetricSaver()
        self.best_val_dice: float = 0.0
        self.store_checkpoints_in_temp_storage: bool = store_checkpoints_in_temp_storage
        self.focal_param_strategy: FocalParamStrategy
        self.save_dir: Path

    def train(
        self,
        dataset_dir: Path,
        total_samples: int = -1,
        epochs: int = 5,
        device: str = "cpu",
        gamma: Optional[float] = None,
        alpha: Optional[float] = None,
    ):
        train_loader, val_loader = create_dataloaders(
            self.data_config, dataset_dir, total_samples
        )

        self._pre_training_preparation(train_loader, val_loader, device, gamma, alpha)

        print(f"Training on device: {device}")
        start = time.time()
        for epoch in range(epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, train_dice, train_per_class_dice = self._train_one_epoch(
                train_loader, device
            )
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")

            val_loss, val_dice, val_per_class_dice = self._validate(val_loader, device)
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

            if epoch % 5 == 0:
                self.graph_maker.save_slice(epoch, train_dice)

            self.metric_saver.save(
                epoch, train_dice, train_per_class_dice, val_dice, val_per_class_dice
            )
            self._save_best_model_if_possible(epoch, val_dice)

        self.graph_maker.make_summary_of_results(epochs)
        train_time = int(time.time() - start)
        print(f"Training completed in {timedelta(seconds=train_time)}")

    def _pre_training_preparation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        gamma: Optional[float],
        alpha: Optional[float],
    ):
        class_distribution = ClassDistributionAnalyzer(train_loader, val_loader)
        class_distribution.print_probability_of_each_class()
        self.metric_saver.save_txt_file(
            class_distribution.get_class_distribution_text(),
            filename="class_distribution.txt",
        )

        if gamma is not None and alpha is not None:
            self.focal_param_strategy = FocalParamStrategy(
                class_distribution.proportions,
                self.metric_saver,
                gamma,
                explicite_alpha=alpha,
            )
        else:
            self.focal_param_strategy = FocalParamStrategy(
                class_distribution.proportions, self.metric_saver
            )

        if CLUSTER_TRAINING_ENABLED or SAVE_BEST_MODEL:
            if self.store_checkpoints_in_temp_storage:
                self.save_dir = get_temp_storage_path() / CURR_RUN
            else:
                self.save_dir = self.data_config["save_dir"] / CURR_RUN

            print(f"[INFO] Save dir for checkpoints: {self.save_dir}")
            self.save_dir.mkdir(exist_ok=True, parents=True)

        self.best_val_dice = 0.0

        if device.startswith("cuda") and torch.cuda.device_count() > 1:
            print(f"\nUsing {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(self.model)

        self.model = self.model.to(device)

    def _focal_loss(
        self,
        logits: torch.Tensor,  # (B, C, D, H, W)
        target: torch.Tensor,  # (B, 1, D, H, W) containing class indices (dtype long)
        params: FocalParamStrategy,  # Contains alpha, gamma and epsilon
    ) -> torch.Tensor:
        """
        Multi-class Focal Loss for imbalanced datasets. Source: https://arxiv.org/pdf/1708.02002
        """
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)
        c = probs.shape[1]

        tgt = target.squeeze(1).long()  # (B, D, H, W)
        one_hot = (
            F.one_hot(tgt, num_classes=c).permute(0, 4, 1, 2, 3).float()
        )  # (B, C, D, H, W)

        pt = (probs * one_hot).sum(dim=1)  # Probability of the true class (B, D, H, W)
        log_pt = torch.log(pt + params.EPSILON)
        focal_term = (
            1 - pt
        ) ** params.gamma  # Focusing term to emphasize hard examples

        alpha_t = torch.tensor(params.alpha, dtype=logits.dtype, device=logits.device)
        alpha_t = alpha_t[tgt]

        loss = -alpha_t * focal_term * log_pt
        return loss.mean()

    def _dice_coefficient_per_class(
        self,
        logits: torch.Tensor,  # (B, C, D, H, W)
        target: torch.Tensor,  # (B, 1, D, H, W) containing class indices (int)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Multi-class Dice on argmax predictions. Returns a dice_per_class."""
        preds = torch.argmax(logits, dim=1)  # (B, D, H, W)
        c = logits.shape[1]
        tgt = target.squeeze(1).long()  # (B, D, H, W)

        pred = F.one_hot(preds, num_classes=c).permute(0, 4, 1, 2, 3).float()
        tgt = F.one_hot(tgt, num_classes=c).permute(0, 4, 1, 2, 3).float()

        dims = (0, 2, 3, 4)

        intersection = (pred * tgt).sum(dim=dims)
        denom = pred.sum(dim=dims) + tgt.sum(dim=dims)

        dice_per_class = (2.0 * intersection + eps) / (denom + eps)

        return dice_per_class

    def _train_one_epoch(
        self,
        dataloader: DataLoader,
        device: str,
    ) -> Tuple[float, float, list[float]]:
        self.model.train()
        epoch_loss: float = 0.0
        dice_scores_per_class: List[torch.Tensor] = []

        with tqdm(dataloader, desc="Training") as progress:
            for i, (images, masks) in enumerate(progress):
                images, masks = prepare_images_and_masks(images, masks, device)

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = self._focal_loss(outputs, masks, self.focal_param_strategy)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    dice_per_class = self._dice_coefficient_per_class(outputs, masks)
                    dice_scores_per_class.append(dice_per_class)
                    epoch_loss += loss.item()

                if i == 0:
                    self.graph_maker.set_prediction_and_ground_truth_slice(
                        outputs, masks
                    )

                progress.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "dice": f"{dice_per_class.mean().item():.4f}",
                    }
                )

        mean_per_class_dices: list = (
            torch.stack(dice_scores_per_class).mean(dim=0).cpu().numpy().tolist()
        )
        mean_dice: float = float(np.mean(mean_per_class_dices))

        return epoch_loss / max(1, len(dataloader)), mean_dice, mean_per_class_dices

    def _validate(
        self,
        dataloader: DataLoader,
        device: str,
    ) -> Tuple[float, float, list[float]]:
        self.model.eval()
        val_loss: float = 0.0
        dice_scores_per_class: List[torch.Tensor] = []

        with torch.no_grad():
            with tqdm(dataloader, desc="Validation") as progress:
                for _, (images, masks) in enumerate(progress):
                    images, masks = prepare_images_and_masks(images, masks, device)

                    outputs = self.model(images)
                    loss = self._focal_loss(outputs, masks, self.focal_param_strategy)
                    val_loss += loss.item()

                    dice_per_class = self._dice_coefficient_per_class(outputs, masks)
                    dice_scores_per_class.append(dice_per_class)

                    progress.set_postfix(
                        {
                            "val_loss": f"{loss.item():.4f}",
                            "val_dice": f"{dice_per_class.mean().item():.4f}",
                        }
                    )

        mean_per_class_dices: list = (
            torch.stack(dice_scores_per_class).mean(dim=0).cpu().numpy().tolist()
        )
        mean_dice: float = float(np.mean(mean_per_class_dices))

        return val_loss / max(1, len(dataloader)), mean_dice, mean_per_class_dices

    def _save_best_model_if_possible(
        self,
        epoch: int,
        val_dice: float,
    ):
        save_config_enabled: bool = CLUSTER_TRAINING_ENABLED or SAVE_BEST_MODEL

        if val_dice > self.best_val_dice and save_config_enabled:
            self.best_val_dice = val_dice
            save_model(
                save_dir=self.save_dir,
                epoch=epoch,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                val_dice=val_dice,
            )
            print(f"Saved best model with val_dice score: {val_dice:.4f}")
