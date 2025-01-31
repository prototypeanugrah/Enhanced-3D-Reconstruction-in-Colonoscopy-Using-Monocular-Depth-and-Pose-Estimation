"""
This script is the PyTorch Lightning module for the DepthAnythingV2 model. It
contains the model definition, training, validation, and testing steps. The
module also includes the SiLogLoss loss function and evaluation metrics.
"""

from typing import Literal

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch
import torchmetrics
import lightning as pl

from Depth_Anything_V2.metric_depth.depth_anything_v2 import dpt
from eval import evaluation


# class SiLogLoss(nn.Module):
#     def __init__(self, lambd=0.5):
#         """
#         Initialize the SiLogLoss loss function.

#         Args:
#             lambd (float, optional): The lambda parameter for the loss function.
#             Defaults to 0.5.
#         """
#         super().__init__()
#         self.lambd = lambd

#     def forward(
#         self,
#         pred: torch.Tensor,
#         target: torch.Tensor,
#         valid_mask: torch.Tensor,
#     ) -> torch.Tensor:
#         """
#         Compute the SiLogLoss loss function.

#         Args:
#             pred (torch.Tensor): The predicted depth map tensor.
#             target (torch.Tensor): The target depth map tensor.
#             valid_mask (torch.Tensor): The valid mask tensor.

#         Returns:
#             torch.Tensor: The computed loss value.
#         """

#         assert pred.dim() in (3, 4), f"Pred should be 3D or 4D, got shape {pred.shape}"
#         assert target.dim() == 4, f"Target should be 4D, got shape {target.shape}"
#         assert valid_mask.dim() == 4, f"Mask should be 4D, got shape {valid_mask.shape}"

#         # Ensure pred and target have the same shape
#         if pred.dim() == 3:  # If pred is [B, H, W]
#             pred = pred.unsqueeze(1)  # Make it [B, 1, H, W]

#         # Resize valid_mask to match pred dimensions
#         if valid_mask.shape[2:] != pred.shape[2:]:
#             valid_mask = torch.nn.functional.interpolate(
#                 valid_mask.float(),
#                 size=pred.shape[2:],
#                 mode="nearest",
#             )

#         valid_mask = valid_mask.bool()
#         valid_mask = valid_mask.detach()

#         diff_log = torch.log(target[valid_mask].flatten()) - torch.log(
#             pred[valid_mask].flatten()
#         )
#         loss = torch.sqrt(
#             torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2)
#         )

#         return loss


class SiLogLoss(nn.Module):
    """Scale-invariant logarithmic loss function for depth estimation."""

    def __init__(self, lambd: float = 0.5):
        """
        Initialize the SiLogLoss loss function.

        Args:
        lambd (float, optional): The lambda parameter for the loss function.
        Defaults to 0.5.
        """
        super().__init__()
        self.lambd = lambd

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(
            torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2)
        )

        return loss


class DepthAnythingV2Module(pl.LightningModule):
    """
    Module for depth estimation using the DepthAnythingV2 model.

    Args:
        encoder (Literal["vits", "vitb", "vitl", "vitg"]): The encoder to use
        in the model.
        min_depth (float, optional): The minimum depth value to clamp the
        output to.
        max_depth (float, optional): The maximum depth value to clamp the
        output to.
        pct_start (float, optional): The percentage of steps to increase the
        learning rate. Warms up the learning rate from 0 to the initial learning
        rate.
        encoder_lr (float, optional): The learning rate for the encoder.
        decoder_lr (float, optional): The learning rate for the decoder.
        max_encoder_lr (float, optional): The maximum learning rate for the
        encoder.
        max_decoder_lr (float, optional): The maximum learning rate for the
        decoder.
    """

    def __init__(
        self,
        encoder: Literal["vits", "vitb", "vitl", "vitg"],
        min_depth: float,
        max_depth: float,
        pct_start: float,
        div_factor: float,
        cycle_momentum: float,
        encoder_lr: float,
        decoder_lr: float,
        # max_encoder_lr: float,
        # max_decoder_lr: float,
        **kwargs,
    ):
        """
        Initialize the DepthEstimationModule.
        """
        super().__init__()

        self.model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
            "vitg": {
                "encoder": "vitg",
                "features": 384,
                "out_channels": [1536, 1536, 1536, 1536],
            },
        }

        self.save_hyperparameters()  # Save hyperparameters for logging

        pretrained_from = (
            f"./base_checkpoints/depth_anything_v2_metric_hypersim_{encoder}.pth"
        )
        self.model = dpt.DepthAnythingV2(
            **{
                **self.model_configs[encoder],
                "max_depth": self.hparams.max_depth,
            }
        )

        # Enable gradient checkpointing after model initialization
        if hasattr(self.model, "encoder") and hasattr(
            self.model.encoder, "set_grad_checkpointing"
        ):
            self.model.encoder.set_grad_checkpointing(enable=True)

        # Load pretrained weights
        self.model.load_state_dict(
            {
                k: v
                for k, v in torch.load(
                    pretrained_from,
                    map_location="cpu",
                ).items()
                if "pretrained" in k
            },
            strict=False,
        )

        # self.loss = nn.MSELoss()
        self.loss = SiLogLoss()

        self.metric = torchmetrics.MetricCollection(
            {
                "d1": torchmetrics.MeanMetric(),
                "abs_rel": torchmetrics.MeanMetric(),
                "rmse": torchmetrics.MeanMetric(),
                "l1": torchmetrics.MeanMetric(),
            }
        )

    def _preprocess_batch(
        self,
        batch: dict,
    ) -> tuple:
        """
        Perform preprocessing on the input batch.

        Args:
            batch (dict): A dict containing the input image and the target

        Returns:
            torch.Tensor: The preprocessed image, depth, and mask tensors
        """
        img, depth = batch["image"], batch["depth"]
        # img, depth = batch["image"], batch["depth"]

        # valid_mask = (
        #     (mask == 1)
        #     & (depth >= self.hparams.min_depth)
        #     & (depth <= self.hparams.max_depth)
        # )

        # # Clamp depth values to min and max values
        # depth = torch.clamp(
        #     depth,
        #     min=self.hparams.min_depth,
        #     max=self.hparams.max_depth,
        # )
        return img, depth  # , valid_mask

    def training_step(
        self,
        batch: dict,
    ) -> torch.Tensor:
        """
        Perform a training step.

        Args:
            batch (dict): A dict containing the input image and the target

        Returns:
            torch.Tensor: The loss value for the training step
        """
        img, depth = self._preprocess_batch(batch)
        # img, depth, valid_mask = self._preprocess_batch(batch)

        pred = self.model(img)
        pred = pred.unsqueeze(1)  # Shape: [B, 1, H, W]

        # Free some memory before continuing
        torch.cuda.empty_cache()

        # pred = pred.clamp(
        #     self.hparams.min_depth,
        #     self.hparams.max_depth,
        # )
        # pred = pred[:, None].clamp(
        #     self.hparams.min_depth,
        #     self.hparams.max_depth,
        # )

        valid_mask = (depth >= self.hparams.min_depth) & (
            depth <= self.hparams.max_depth
        )

        loss = self.loss(
            pred,
            depth,
            valid_mask,
        )
        self.log(
            "train_loss",
            loss,
            batch_size=img.shape[0],
        )

        # with torch.no_grad():
        # Compute and log evaluation metrics
        metrics = evaluation.compute_errors(
            pred[valid_mask].detach().flatten(),
            depth[valid_mask].detach().flatten(),
            # pred[depth > 1e-4].flatten(),
            # depth[depth > 1e-4].flatten(),
        )
        for metric_name, value in metrics.items():
            self.metric[metric_name](value)
            self.log(
                f"Train/train_{metric_name}",
                value.item() if torch.is_tensor(value) else value,
                batch_size=img.shape[0],
            )

        # Clear cache if needed
        # del pred
        torch.cuda.empty_cache()

        return loss

    def validation_step(
        self,
        batch: dict,
        # batch_idx: int,
    ) -> torch.Tensor:
        """
        Perform a validation step.

        Args:
            batch (dict): A dict containing the input image and the target
            batch_idx (int): The index of the batch

        Returns:
            torch.Tensor: The loss value for the validation step
        """
        img, depth = self._preprocess_batch(batch)
        # img, depth, valid_mask = self._preprocess_batch(batch)

        pred = self.model(img)
        pred = pred.unsqueeze(1)  # Shape: [B, 1, H, W]

        # Clear cache if needed
        torch.cuda.empty_cache()

        # pred = pred.clamp(
        #     self.hparams.min_depth,
        #     self.hparams.max_depth,
        # )
        # pred = pred[:, None].clamp(
        #     self.hparams.min_depth,
        #     self.hparams.max_depth,
        # )

        valid_mask = (depth >= self.hparams.min_depth) & (
            depth <= self.hparams.max_depth
        )

        # loss = self.loss(pred, depth)
        loss = self.loss(
            pred,
            depth,
            valid_mask,
        )
        self.log(
            "val_loss",
            loss,
            batch_size=img.shape[0],
        )

        with torch.no_grad():
            # Compute and log evaluation metrics
            metrics = evaluation.compute_errors(
                pred[valid_mask].detach().flatten(),
                depth[valid_mask].detach().flatten(),
                # pred[depth > 1e-4].flatten(),
                # depth[depth > 1e-4].flatten(),
            )
            for metric_name, value in metrics.items():
                self.metric[metric_name](value)
                self.log(
                    f"Val/val_{metric_name}",
                    value.item() if torch.is_tensor(value) else value,
                    prog_bar=True,
                    batch_size=img.shape[0],
                )

        # Clear cache if needed
        # del pred
        torch.cuda.empty_cache()

        return loss

    def on_test_epoch_start(self):
        # Reset all metrics at the start of the test epoch
        self.metric.reset()

    def test_step(
        self,
        batch: dict,
    ) -> dict:
        """
        Perform a test step.

        Args:
            batch (dict): A dict containing the input image and the target
        """
        img, depth = self._preprocess_batch(batch)
        # img, depth, valid_mask = self._preprocess_batch(batch)

        pred = self.model(img)
        pred = pred.unsqueeze(1)  # Shape: [B, 1, H, W]
        # pred = pred.clamp(
        #     self.hparams.min_depth,
        #     self.hparams.max_depth,
        # )

        valid_mask = (depth >= self.hparams.min_depth) & (
            depth <= self.hparams.max_depth
        )

        with torch.no_grad():
            # Compute and log evaluation metrics
            metrics = evaluation.compute_errors(
                pred[valid_mask].detach().flatten(),
                depth[valid_mask].detach().flatten(),
                # pred[depth > 1e-4].flatten(),
                # depth[depth > 1e-4].flatten(),
            )

            for metric_name, value in metrics.items():
                self.metric[metric_name](
                    value.item() if torch.is_tensor(value) else value
                )

        # Clear cache if needed
        # del pred
        torch.cuda.empty_cache()

        # Return a dictionary of metrics
        return {
            "d1": metrics.get("d1", 0.0),
            "abs_rel": metrics.get("abs_rel", 0.0),
            "rmse": metrics.get("rmse", 0.0),
            "l1": metrics.get("l1", 0.0),
        }

    def on_test_epoch_end(self):
        # Compute final metrics
        final_metrics = self.metric.compute()

        # Log the final computed metrics
        for metric_name, value in final_metrics.items():
            self.log(f"Test/test_{metric_name}", value)

        # Reset metrics after computing
        self.metric.reset()

    def predict_step(
        self,
        batch: dict,
    ) -> torch.Tensor:
        """
        Perform a prediction step.

        Args:
            batch (dict): A dict containing the input image and the target

        Returns:
            torch.Tensor: The predicted depth map
        """
        img, _ = self._preprocess_batch(batch)
        # img, _, _ = self._preprocess_batch(batch)

        pred = self.model(img)
        # pred = pred.clamp(
        #     self.hparams.min_depth,
        #     self.hparams.max_depth,
        # )
        # pred = pred[:, None].clamp(
        #     self.hparams.min_depth,
        #     self.hparams.max_depth,
        # )

        return pred

    def configure_optimizers(self) -> dict:

        # Define optimizer with parameter groups
        optimizer = optim.AdamW(
            [
                {
                    "params": [
                        param
                        for name, param in self.named_parameters()
                        if "pretrained" in name
                    ],
                    "lr": self.hparams.encoder_lr,  # Encoder learning rate
                    "name": "encoder_lr",
                },
                {
                    "params": [
                        param
                        for name, param in self.named_parameters()
                        if "pretrained" not in name
                    ],
                    "lr": self.hparams.decoder_lr,  # Decoder learning rate
                    "name": "decoder_lr",
                },
            ],
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Define LR scheduler for each parameter group
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=[
                self.hparams.encoder_lr,
                self.hparams.decoder_lr,
            ],
            pct_start=self.hparams.pct_start,
            div_factor=self.hparams.div_factor,
            cycle_momentum=self.hparams.cycle_momentum,
            # pct_start=0.05,
            # cycle_momentum=False,
            # div_factor=1e9,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "lr_scheduler",
            },
        }
