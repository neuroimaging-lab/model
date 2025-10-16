from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation.config import CURR_RUN, SAVE_BEST_MODEL, SUPER_COMPUTER_ENABLED
from segmentation.focal_param_strategy import FocalParamStrategy
from segmentation.train_utils import (
    create_dataloaders,
    prepare_images_and_masks,
    save_model,
)
from util.class_distribution_analyzer import ClassDistributionAnalyzer
from util.graph_maker import GraphMaker
from util.metric_saver import MetricSaver


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
        include_bg: bool = False,  # Background (usually dominant) can negatively impact Dice score
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

        self.include_bg = include_bg
        self.graph_maker = GraphMaker()
        self.metric_saver = MetricSaver()
        self.best_val_dice: float = 0.0
        self.focal_param_strategy: FocalParamStrategy
        self.save_dir: Path

    def train(
        self,
        dataset_dir: Path,
        total_samples: int = -1,
        epochs: int = 5,
        device: str = "cpu",
        checkpoints_enabled: bool = False,
    ):
        train_loader, val_loader = create_dataloaders(
            self.data_config, dataset_dir, total_samples
        )

        self._pre_training_preparation(
            train_loader, val_loader, checkpoints_enabled, device
        )

        print(f"Training on device: {device}")
        start = time.time()
        for epoch in range(epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")

            train_loss, train_dice = self._train_one_epoch(train_loader, device)
            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")

            if epoch % 5 == 0:
                self.graph_maker.save_slice(epoch, train_dice)

            val_loss, val_dice = self._validate(val_loader, device)
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

            self.metric_saver.save(epoch, train_dice, val_dice)
            self._checkpoints_and_validation(checkpoints_enabled, epoch, val_dice)

        self.graph_maker.make_summary_of_results(epochs)
        train_time = int(time.time() - start)
        print(f"Training completed in {timedelta(seconds=train_time)}")

    def _pre_training_preparation(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoints_enabled: bool,
        device: str,
    ):
        class_distribution = ClassDistributionAnalyzer(train_loader, val_loader)
        class_distribution.print_probability_of_each_class()
        self.metric_saver.save_txt_file(
            class_distribution.get_class_distribution_text(),
            filename="class_distribution.txt",
        )

        self.focal_param_strategy = FocalParamStrategy(
            class_distribution.proportions, self.metric_saver
        )

        if SUPER_COMPUTER_ENABLED or SAVE_BEST_MODEL or checkpoints_enabled:
            self.save_dir: Path = self.data_config["save_dir"] / CURR_RUN
            self.save_dir.mkdir(exist_ok=True, parents=True)

        self.best_val_dice = 0.0
        self.model = self.model.to(device)

    def _focal_loss(
        self,
        logits: torch.Tensor,  # (B, C, D, H, W)
        target: torch.Tensor,  # (B, 1, D, H, W) containing class indices (dtype long)
        params: FocalParamStrategy,  # contains alpha, gamma and epsilon
    ) -> torch.Tensor:
        """
        Multi-class Focal Loss for imbalanced datasets.
        Args:
            logits: Raw model outputs (logits) of shape (B, C, D, H, W).
            target: Ground truth tensor of shape (B, 1, D, H, W) containing class indices.
            alpha: Balancing factor
            gamma: Focusing parameter to reduce loss contribution from easy examples.
            eps: Small value to avoid division by zero.
        Returns:
            Focal loss value as a single scalar tensor.
        """
        # Normalize logits to probabilities using softmax
        probs = F.softmax(logits, dim=1)  # (B, C, D, H, W)
        c = probs.shape[1]

        # Convert target to one-hot encoding
        tgt = target.squeeze(1).long()  # (B, D, H, W)
        one_hot = (
            F.one_hot(tgt, num_classes=c).permute(0, 4, 1, 2, 3).float()
        )  # (B, C, D, H, W)

        # Compute the focal loss
        pt = (probs * one_hot).sum(dim=1)  # Probability of the true class (B, D, H, W)
        log_pt = torch.log(pt + params.eps)  # Log probability of the true class
        focal_term = (
            1 - pt
        ) ** params.gamma  # Focusing term to emphasize hard examples

        alpha_t = torch.tensor(params.alpha, dtype=logits.dtype, device=logits.device)
        alpha_t = alpha_t[tgt]

        loss = -alpha_t * focal_term * log_pt  # Focal loss formula
        return loss.mean()

    def _dice_coefficient(
        self,
        logits: torch.Tensor,  # (B, C, D, H, W)
        target: torch.Tensor,  # (B, 1, D, H, W) containing class indices (int)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Multi-class Dice on argmax predictions. Returns mean Dice across classes."""
        preds = torch.argmax(logits, dim=1)  # (B, D, H, W)
        c = logits.shape[1]
        tgt = target.squeeze(1).long()  # (B, D, H, W)

        pred = F.one_hot(preds, num_classes=c).permute(0, 4, 1, 2, 3).float()
        tgt = F.one_hot(tgt, num_classes=c).permute(0, 4, 1, 2, 3).float()

        if not self.include_bg and c > 1:
            pred = pred[:, 1:, ...]
            tgt = tgt[:, 1:, ...]

        dims = (0, 2, 3, 4)

        # Here we no longer have floats in pred vectors, now we use argmax, so we have hard predictions
        # and the only way to get non 0 value, is if the class is the same in both pred and tgt
        intersection = (pred * tgt).sum(dim=dims)
        denom = pred.sum(dim=dims) + tgt.sum(dim=dims)

        dice_per_class = (2.0 * intersection + eps) / (denom + eps)

        return dice_per_class.mean()

    def _train_one_epoch(
        self,
        dataloader: DataLoader,
        device: str,
    ) -> Tuple[float, float]:
        self.model.train()
        epoch_loss: float = 0.0
        dice_scores: List[float] = []

        with tqdm(dataloader, desc="Training") as progress:
            for i, (images, masks) in enumerate(progress):
                images, masks = prepare_images_and_masks(images, masks, device)

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = self._focal_loss(outputs, masks, self.focal_param_strategy)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    dice = self._dice_coefficient(outputs, masks)
                    dice_scores.append(dice.item())
                    epoch_loss += loss.item()

                if i == 0:
                    self.graph_maker.set_prediction_and_ground_truth_slice(
                        outputs, masks
                    )

                progress.set_postfix(
                    {"loss": f"{loss.item():.4f}", "dice": f"{dice_scores[-1]:.4f}"}
                )

        return epoch_loss / max(1, len(dataloader)), float(
            np.mean(dice_scores) if dice_scores else 0.0
        )

    def _validate(
        self,
        dataloader: DataLoader,
        device: str,
    ) -> Tuple[float, float]:
        self.model.eval()
        val_loss: float = 0.0
        dice_scores: List[float] = []

        with torch.no_grad():
            with tqdm(dataloader, desc="Validation") as progress:
                for _, (images, masks) in enumerate(progress):
                    images, masks = prepare_images_and_masks(images, masks, device)

                    outputs = self.model(images)
                    loss = self._focal_loss(outputs, masks, self.focal_param_strategy)

                    dice = self._dice_coefficient(outputs, masks)
                    dice_scores.append(dice.item())
                    val_loss += loss.item()

                    progress.set_postfix(
                        {
                            "val_loss": f"{loss.item():.4f}",
                            "val_dice": f"{dice_scores[-1]:.4f}",
                        }
                    )

        return val_loss / max(1, len(dataloader)), float(
            np.mean(dice_scores) if dice_scores else 0.0
        )

    def _checkpoints_and_validation(
        self,
        checkpoints_enabled: bool,
        epoch: int,
        val_dice: float,
    ):
        save_config_enabled: bool = SUPER_COMPUTER_ENABLED or SAVE_BEST_MODEL

        if checkpoints_enabled:
            save_model(
                save_dir=self.save_dir,
                epoch=epoch,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                val_dice=val_dice,
            )

        if val_dice > self.best_val_dice and save_config_enabled:
            self.best_val_dice = val_dice
            save_model(
                save_dir=self.save_dir,
                epoch=epoch,
                model_state_dict=self.model.state_dict(),
                optimizer_state_dict=self.optimizer.state_dict(),
                val_dice=val_dice,
                is_best_model=True,
            )
            print(f"Saved best model with val_dice score: {val_dice:.4f}")
