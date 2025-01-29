"""
This script is the PyTorch Lightning module for the combined DepthAnythingV2 
model. This module trains the model on the combined SimCol and C3VD datasets.
It contains the model definition, training, validation, and testing steps. The
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
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
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
        simcol_max_depth: float,
        c3vd_max_depth: float,
        pct_start: float,
        div_factor: float,
        cycle_momentum: float,
        encoder_lr: float,
        decoder_lr: float,
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
        }

        self.save_hyperparameters()  # Save hyperparameters for logging

        # Store dataset-specific max depths
        self.simcol_max_depth = simcol_max_depth  # in cm
        self.c3vd_max_depth = c3vd_max_depth / 10  # convert mm to cm

        # Use the larger max_depth for model initialization (both in cm)
        model_max_depth = max(self.simcol_max_depth, self.c3vd_max_depth)

        pretrained_from = (
            f"./base_checkpoints/depth_anything_v2_metric_hypersim_{encoder}.pth"
        )
        self.model = dpt.DepthAnythingV2(
            **{
                **self.model_configs[encoder],
                "max_depth": model_max_depth,
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

        # Create separate metric collections for each dataset
        self.simcol_metrics = torchmetrics.MetricCollection(
            {
                "d1": torchmetrics.MeanMetric(),
                "abs_rel": torchmetrics.MeanMetric(),
                "rmse": torchmetrics.MeanMetric(),
                "l1": torchmetrics.MeanMetric(),
            },
            prefix="simcol_",
        )

        self.c3vd_metrics = torchmetrics.MetricCollection(
            {
                "d1": torchmetrics.MeanMetric(),
                "abs_rel": torchmetrics.MeanMetric(),
                "rmse": torchmetrics.MeanMetric(),
                "l1": torchmetrics.MeanMetric(),
            },
            prefix="c3vd_",
        )

    def _preprocess_batch(
        self,
        batch: dict,
    ) -> tuple:
        img, depth = batch["image"], batch["depth"]
        # img, depth, mask = batch["image"], batch["depth"], batch["mask"]

        # Convert source list to tensor if it isn't already
        if isinstance(batch["source"], list):
            source = torch.tensor(
                [1 if s == "c3vd" else 0 for s in batch["source"]],
                device=img.device,
            )
        else:
            source = batch["source"]

        # Ensure source tensor has the same shape as depth for broadcasting
        source = source.view(-1, 1, 1, 1).expand_as(depth)

        # Convert C3VD depths from mm to cm to match SimCol
        c3vd_mask = source == 1  # Using binary comparison instead of string comparison
        if c3vd_mask.any():
            depth[c3vd_mask] = depth[c3vd_mask] / 10  # mm to cm conversion

        # # Create valid mask based on dataset source
        # valid_mask = mask == 1
        # valid_mask = valid_mask & (
        #     torch.where(
        #         source == 0,
        #         (depth >= self.hparams.min_depth) & (depth <= self.simcol_max_depth),
        #         (depth >= self.hparams.min_depth) & (depth <= self.c3vd_max_depth),
        #     )
        # )

        # Clamp depth values to min and max values (all in cm)
        # depth = torch.where(
        #     source == 0,
        #     torch.clamp(depth, self.hparams.min_depth, self.simcol_max_depth),
        #     torch.clamp(depth, self.hparams.min_depth, self.c3vd_max_depth),
        # )

        return (
            img,
            depth,
            # valid_mask,
            source,
        )

    def _clamp_predictions(self, pred, source):
        """
        Clamp predictions based on the dataset source.
        All operations are done in centimeters.

        Args:
            pred (torch.Tensor): Predictions to clamp
            source (str): Dataset source ('simcol' or 'c3vd')

        Returns:
            torch.Tensor: Clamped predictions in appropriate units
        """
        if source == "simcol":
            return pred.clamp(
                self.hparams.min_depth,
                self.simcol_max_depth,
            )
        else:  # c3vd
            return pred.clamp(
                self.hparams.min_depth,
                self.c3vd_max_depth,
            )

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

        def print_mem(step: str):
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            print(f"\n{step}:")
            print(f"Allocated: {allocated:.2f}GB")
            print(f"Reserved: {reserved:.2f}GB")
            print(f"Max Allocated: {max_allocated:.2f}GB")

        # print_mem("Start of training step")

        img, depth, source = self._preprocess_batch(batch)
        # img, depth, valid_mask, source = self._preprocess_batch(batch)
        # print_mem("After preprocessing")

        pred = self.model(img)
        # print_mem("After model forward")
        # pred = pred[:, None]  # Add channel dimension

        # Free some memory before continuing
        torch.cuda.empty_cache()

        # Create masks for each dataset
        simcol_mask = source == 0
        c3vd_mask = source == 1

        valid_mask = torch.where(
            source == 0,
            (depth >= self.hparams.min_depth) & (depth <= self.simcol_max_depth),
            (depth >= self.hparams.min_depth) & (depth <= self.c3vd_max_depth),
        )

        # with torch.no_grad():
        # Apply appropriate clamping based on dataset source
        # pred = torch.where(
        #     simcol_mask,
        #     self._clamp_predictions(pred, "simcol"),
        #     self._clamp_predictions(pred, "c3vd"),
        # )

        # Calculate loss using valid mask
        loss = self.loss(pred, depth, valid_mask)
        # print_mem("After loss computation")
        self.log(
            "train_loss",
            loss,
            batch_size=img.shape[0],
        )

        # if self.training:
        with torch.no_grad():
            # Handle SimCol metrics (already in cm)
            if simcol_mask.any():
                simcol_metrics = evaluation.compute_errors(
                    pred[simcol_mask & valid_mask].detach().flatten(),
                    depth[simcol_mask & valid_mask].detach().flatten(),
                )
                # print_mem("After SimCol metrics")
                for metric_name, value in simcol_metrics.items():
                    self.simcol_metrics[metric_name](
                        value.item() if torch.is_tensor(value) else value
                    )
                    self.log(
                        f"SimCol/train_simcol_{metric_name}",
                        value.item() if torch.is_tensor(value) else value,
                        batch_size=img.shape[0],
                    )

            # Handle C3VD metrics (convert back to mm)
            if c3vd_mask.any():
                c3vd_metrics = evaluation.compute_errors(
                    (pred[c3vd_mask & valid_mask] * 10)
                    .detach()
                    .flatten(),  # convert to mm
                    (depth[c3vd_mask & valid_mask] * 10)
                    .detach()
                    .flatten(),  # convert to mm
                )
                # print_mem("After C3VD metrics")
                for metric_name, value in c3vd_metrics.items():
                    self.c3vd_metrics[metric_name](
                        value.item() if torch.is_tensor(value) else value
                    )
                    self.log(
                        f"C3VD/train_c3vd_{metric_name}",
                        value.item() if torch.is_tensor(value) else value,
                        batch_size=img.shape[0],
                    )

        # Clear cache after all computations
        del pred
        torch.cuda.empty_cache()

        return loss

    def validation_step(
        self,
        batch: dict,
    ) -> torch.Tensor:
        """
        Perform a validation step.

        Args:
            batch (dict): A dict containing the input image and the target

        Returns:
            torch.Tensor: The loss value for the validation step
        """
        img, depth, source = self._preprocess_batch(batch)
        # img, depth, valid_mask, source = self._preprocess_batch(batch)

        pred = self.model(img)
        # pred = pred[:, None]  # Add channel dimension

        # Create masks for each dataset
        simcol_mask = source == 0
        c3vd_mask = source == 1

        # Apply appropriate clamping based on dataset source
        # pred = torch.where(
        #     simcol_mask,
        #     self._clamp_predictions(pred, "simcol"),
        #     self._clamp_predictions(pred, "c3vd"),
        # )

        valid_mask = torch.where(
            source == 0,
            (depth >= self.hparams.min_depth) & (depth <= self.simcol_max_depth),
            (depth >= self.hparams.min_depth) & (depth <= self.c3vd_max_depth),
        )

        # Calculate loss using valid mask
        loss = self.loss(pred, depth, valid_mask)
        self.log(
            "val_loss",
            loss,
            batch_size=img.shape[0],
        )

        with torch.no_grad():
            # Handle SimCol metrics (already in cm)
            if simcol_mask.any():
                simcol_metrics = evaluation.compute_errors(
                    pred[simcol_mask & valid_mask].detach().flatten(),
                    depth[simcol_mask & valid_mask].detach().flatten(),
                )
                for metric_name, value in simcol_metrics.items():
                    self.simcol_metrics[metric_name](value)
                    self.log(
                        f"SimCol/val_simcol_{metric_name}",
                        value.item() if torch.is_tensor(value) else value,
                        prog_bar=True,
                        batch_size=img.shape[0],
                    )

            # Handle C3VD metrics (convert back to mm)
            if c3vd_mask.any():
                c3vd_metrics = evaluation.compute_errors(
                    (pred[c3vd_mask & valid_mask] * 10)
                    .detach()
                    .flatten(),  # convert to mm
                    (depth[c3vd_mask & valid_mask] * 10)
                    .detach()
                    .flatten(),  # convert to mm
                )
                for metric_name, value in c3vd_metrics.items():
                    self.c3vd_metrics[metric_name](value)
                    self.log(
                        f"C3VD/val_c3vd_{metric_name}",
                        value.item() if torch.is_tensor(value) else value,
                        prog_bar=True,
                        batch_size=img.shape[0],
                    )

        # Clear cache after all computations
        del pred
        torch.cuda.empty_cache()

        return loss

    def on_test_epoch_start(self):
        """Reset metrics at the start of test epoch."""
        self.simcol_metrics.reset()
        self.c3vd_metrics.reset()

    def test_step(
        self,
        batch: dict,
        batch_idx: int,
    ) -> None:
        """
        Perform a test step.

        Args:
            batch (dict): A dict containing the input image and the target
            batch_idx (int): The index of the batch
        """
        img, depth, source = self._preprocess_batch(batch)
        # img, depth, valid_mask, source = self._preprocess_batch(batch)

        pred = self.model(img)
        # pred = pred[:, None]  # Add channel dimension

        # Create masks for each dataset
        simcol_mask = source == 0
        c3vd_mask = source == 1

        # Apply appropriate clamping based on dataset source
        # pred = torch.where(
        #     simcol_mask,
        #     self._clamp_predictions(pred, "simcol"),
        #     self._clamp_predictions(pred, "c3vd"),
        # )

        valid_mask = torch.where(
            source == 0,
            (depth >= self.hparams.min_depth) & (depth <= self.simcol_max_depth),
            (depth >= self.hparams.min_depth) & (depth <= self.c3vd_max_depth),
        )

        with torch.no_grad():
            # Handle SimCol metrics (already in cm)
            if simcol_mask.any():
                simcol_metrics = evaluation.compute_errors(
                    pred[simcol_mask & valid_mask].detach().flatten(),
                    depth[simcol_mask & valid_mask].detach().flatten(),
                )
                for metric_name, value in simcol_metrics.items():
                    self.simcol_metrics[metric_name](
                        value.item() if torch.is_tensor(value) else value
                    )

            # Handle C3VD metrics (convert back to mm)
            if c3vd_mask.any():
                c3vd_metrics = evaluation.compute_errors(
                    (pred[c3vd_mask & valid_mask] * 10)
                    .detach()
                    .flatten(),  # convert to mm
                    (depth[c3vd_mask & valid_mask] * 10)
                    .detach()
                    .flatten(),  # convert to mm
                )
                for metric_name, value in c3vd_metrics.items():
                    self.c3vd_metrics[metric_name](
                        value.item() if torch.is_tensor(value) else value
                    )
        # Clear cache after all computations
        torch.cuda.empty_cache()

    def on_test_epoch_end(self):
        """Compute and log final metrics at the end of test epoch."""
        # Compute final metrics for both datasets
        final_simcol_metrics = self.simcol_metrics.compute()
        final_c3vd_metrics = self.c3vd_metrics.compute()

        # Log the final computed metrics
        for metric_name, value in final_simcol_metrics.items():
            self.log(
                f"SimCol/test_{metric_name}",
                value.item() if torch.is_tensor(value) else value,
            )

        for metric_name, value in final_c3vd_metrics.items():
            self.log(
                f"C3VD/test_{metric_name}",
                value.item() if torch.is_tensor(value) else value,
            )

        # Reset metrics after computing
        self.simcol_metrics.reset()
        self.c3vd_metrics.reset()

    def predict_step(
        self,
        batch: dict,
    ) -> torch.Tensor:
        """
        Perform a prediction step.

        Args:
            batch (dict): A dict containing the input image and the target

        Returns:
            torch.Tensor: The predicted depth map, with appropriate clamping based on dataset source
            and units (cm for SimCol, mm for C3VD)
        """
        img, _, _ = self._preprocess_batch(batch)
        # img, _, valid_mask, source = self._preprocess_batch(batch)

        pred = self.model(img)
        # pred = pred[:, None]  # Add channel dimension

        # # Create masks for each dataset
        # simcol_mask = source == 0
        # c3vd_mask = source == 1

        # Apply appropriate clamping based on dataset source
        # pred = torch.where(
        #     simcol_mask,
        #     self._clamp_predictions(pred, "simcol"),  # SimCol predictions stay in cm
        #     self._clamp_predictions(pred, "c3vd")
        #     * 10,  # C3VD predictions converted to mm
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
                },
                {
                    "params": [
                        param
                        for name, param in self.named_parameters()
                        if "pretrained" not in name
                    ],
                    "lr": self.hparams.decoder_lr,  # Decoder learning rate
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
            # div_factor=1e9,
            # pct_start=0.05,
            # cycle_momentum=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
