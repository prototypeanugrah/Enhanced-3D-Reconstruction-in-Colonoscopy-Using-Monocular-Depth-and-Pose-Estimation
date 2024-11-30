"Module for Depth Estimation Model Config"

from typing import Literal

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch
import torchmetrics
import lightning as pl

from Depth_Anything_V2.metric_depth.depth_anything_v2 import dpt
from eval import evaluation


class DepthAnythingV2Module(pl.LightningModule):
    """
    Module for depth estimation using the DepthAnythingV2 model.

    Args:
        encoder (Literal["vits", "vitb", "vitl", "vitg"]): The encoder to use
        in the model.
        min_depth (float, optional): The minimum depth value to clamp the
        output to. Defaults to 1e-4.
        max_depth (float, optional): The maximum depth value to clamp the
        output to. Defaults to 20.0.
        lr (float, optional): The learning rate to use for training. Defaults
        to 5e-6.
    """

    def __init__(
        self,
        encoder: Literal["vits", "vitb", "vitl", "vitg"],
        min_depth: float = 1e-4,
        simcol_max_depth: float = 20.0,
        c3vd_max_depth: float = 100.0,
        lr: float = 5e-6,
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

        dataset = "hypersim"
        pretrained_from = (
            f"./base_checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth"
        )
        self.model = dpt.DepthAnythingV2(
            **{
                **self.model_configs[encoder],
                "max_depth": model_max_depth,
            }
        )

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

        self.loss = nn.MSELoss()

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

        # Clamp depth values to min and max values (all in cm)
        depth = torch.where(
            source == 0,
            torch.clamp(depth, self.hparams.min_depth, self.simcol_max_depth),
            torch.clamp(depth, self.hparams.min_depth, self.c3vd_max_depth),
        )

        return img, depth

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
        img, depth = self._preprocess_batch(batch)

        pred = self.model(img)
        pred = pred[:, None]  # Add channel dimension

        # Convert source list to tensor if it isn't already
        if isinstance(batch["source"], list):
            source = torch.tensor(
                [1 if s == "c3vd" else 0 for s in batch["source"]],
                device=img.device,
            )
        else:
            source = batch["source"]

        # Create masks for each dataset
        simcol_mask = source == 0
        c3vd_mask = source == 1

        # Handle SimCol predictions (already in cm)
        if simcol_mask.any():
            simcol_pred = self._clamp_predictions(pred[simcol_mask], "simcol")
            simcol_metrics = evaluation.compute_errors(
                simcol_pred[depth[simcol_mask] > 1e-4].flatten(),
                depth[simcol_mask][depth[simcol_mask] > 1e-4].flatten(),
            )
            for metric_name, value in simcol_metrics.items():
                self.simcol_metrics[metric_name](value)
                self.log(
                    f"SimCol/train_simcol_{metric_name}",
                    value,
                    batch_size=img.shape[0],
                )

        # Handle C3VD predictions (convert back to mm for metrics)
        if c3vd_mask.any():
            c3vd_pred = self._clamp_predictions(pred[c3vd_mask], "c3vd")
            # Convert predictions and ground truth back to mm for evaluation
            c3vd_metrics = evaluation.compute_errors(
                (c3vd_pred * 10)[depth[c3vd_mask] > 1e-4].flatten(),  # convert to mm
                (depth[c3vd_mask] * 10)[
                    depth[c3vd_mask] > 1e-4
                ].flatten(),  # convert to mm
            )
            for metric_name, value in c3vd_metrics.items():
                self.c3vd_metrics[metric_name](value)
                self.log(
                    f"C3VD/train_c3vd_{metric_name}",
                    value,
                    batch_size=img.shape[0],
                )

        # Calculate combined loss using appropriately clamped predictions
        # (all in cm)
        pred_clamped = torch.where(
            simcol_mask.view(-1, 1, 1, 1),
            self._clamp_predictions(pred, "simcol"),
            self._clamp_predictions(pred, "c3vd"),
        )

        loss = self.loss(pred_clamped, depth)
        self.log(
            "train_loss",
            loss,
            batch_size=img.shape[0],
        )

        return loss

    def validation_step(
        self,
        batch: dict,
    ) -> torch.Tensor:
        img, depth = self._preprocess_batch(batch)  # depth now in cm

        pred = self.model(img)
        pred = pred[:, None]  # Add channel dimension

        # Convert source list to tensor if it isn't already
        if isinstance(batch["source"], list):
            source = torch.tensor(
                [1 if s == "c3vd" else 0 for s in batch["source"]],
                device=img.device,
            )
        else:
            source = batch["source"]

        # Create masks for each dataset
        simcol_mask = source == 0
        c3vd_mask = source == 1

        # Handle SimCol predictions (already in cm)
        if simcol_mask.any():
            simcol_pred = self._clamp_predictions(pred[simcol_mask], "simcol")
            simcol_metrics = evaluation.compute_errors(
                simcol_pred[depth[simcol_mask] > 1e-4].flatten(),
                depth[simcol_mask][depth[simcol_mask] > 1e-4].flatten(),
            )
            for metric_name, value in simcol_metrics.items():
                self.simcol_metrics[metric_name](value)
                self.log(
                    f"SimCol/val_simcol_{metric_name}",
                    value,
                    prog_bar=True,
                    batch_size=img.shape[0],
                )

        # Handle C3VD predictions (convert back to mm for metrics)
        if c3vd_mask.any():
            c3vd_pred = self._clamp_predictions(pred[c3vd_mask], "c3vd")
            # Convert predictions and ground truth back to mm for evaluation
            c3vd_metrics = evaluation.compute_errors(
                (c3vd_pred * 10)[depth[c3vd_mask] > 1e-4].flatten(),  # convert to mm
                (depth[c3vd_mask] * 10)[
                    depth[c3vd_mask] > 1e-4
                ].flatten(),  # convert to mm
            )
            for metric_name, value in c3vd_metrics.items():
                self.c3vd_metrics[metric_name](value)
                self.log(
                    f"C3VD/val_c3vd_{metric_name}",
                    value,
                    prog_bar=True,
                    batch_size=img.shape[0],
                )

        # Calculate combined loss using appropriately clamped predictions (all in cm)
        pred_clamped = torch.where(
            simcol_mask.view(-1, 1, 1, 1),
            self._clamp_predictions(pred, "simcol"),
            self._clamp_predictions(pred, "c3vd"),
        )

        loss = self.loss(pred_clamped, depth)
        self.log(
            "val_loss",
            loss,
            batch_size=img.shape[0],
        )

        return loss

    def on_test_epoch_start(self):
        # Reset all metrics at the start of the test epoch
        self.metric.reset()

    def test_step(
        self,
        batch,
        batch_idx,
    ):
        img, depth = self._preprocess_batch(batch)  # depth now in cm

        pred = self.model(img)
        pred = pred[:, None]  # Add channel dimension

        # Convert source list to tensor if it isn't already
        if isinstance(batch["source"], list):
            source = torch.tensor(
                [1 if s == "c3vd" else 0 for s in batch["source"]],
                device=img.device,
            )
        else:
            source = batch["source"]

        # Create masks for each dataset
        simcol_mask = source == 0
        c3vd_mask = source == 1

        # Handle SimCol predictions (already in cm)
        if simcol_mask.any():
            simcol_pred = self._clamp_predictions(pred[simcol_mask], "simcol")
            simcol_metrics = evaluation.compute_errors(
                simcol_pred[depth[simcol_mask] > 1e-4].flatten(),
                depth[simcol_mask][depth[simcol_mask] > 1e-4].flatten(),
            )
            for metric_name, value in simcol_metrics.items():
                self.simcol_metrics[metric_name](value)

        # Handle C3VD predictions (convert back to mm for metrics)
        if c3vd_mask.any():
            c3vd_pred = self._clamp_predictions(pred[c3vd_mask], "c3vd")
            # Convert predictions and ground truth back to mm for evaluation
            c3vd_metrics = evaluation.compute_errors(
                (c3vd_pred * 10)[depth[c3vd_mask] > 1e-4].flatten(),  # convert to mm
                (depth[c3vd_mask] * 10)[
                    depth[c3vd_mask] > 1e-4
                ].flatten(),  # convert to mm
            )
            for metric_name, value in c3vd_metrics.items():
                self.c3vd_metrics[metric_name](value)

    def on_test_epoch_end(self):
        # Compute final metrics for both datasets
        final_simcol_metrics = self.simcol_metrics.compute()
        final_c3vd_metrics = self.c3vd_metrics.compute()

        # Log the final computed metrics
        for metric_name, value in final_simcol_metrics.items():
            self.log(f"SimCol/test_{metric_name}", value)

        for metric_name, value in final_c3vd_metrics.items():
            self.log(f"C3VD/test_{metric_name}", value)

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
        """
        img, _ = self._preprocess_batch(batch)

        pred = self.model(img)
        pred = pred[:, None]  # Add channel dimension

        # Convert source list to tensor if it isn't already
        if isinstance(batch["source"], list):
            source = torch.tensor(
                [1 if s == "c3vd" else 0 for s in batch["source"]],
                device=img.device,
            )
        else:
            source = batch["source"]

        # Create masks for each dataset
        simcol_mask = source == 0
        c3vd_mask = source == 1

        # Apply appropriate clamping based on dataset source
        pred_clamped = torch.zeros_like(pred)
        if simcol_mask.any():
            pred_clamped[simcol_mask] = self._clamp_predictions(
                pred[simcol_mask],
                "simcol",
            )
        if c3vd_mask.any():
            pred_clamped[c3vd_mask] = self._clamp_predictions(
                pred[c3vd_mask],
                "c3vd",
            )
            # Convert C3VD predictions back to mm for final output
            pred_clamped[c3vd_mask] = pred_clamped[c3vd_mask] * 10

        return pred_clamped

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            [
                {
                    "params": [
                        param
                        for name, param in self.named_parameters()
                        if "pretrained" in name
                    ],
                    "lr": self.hparams.lr,  # Encoder learning rate
                },
                {
                    "params": [
                        param
                        for name, param in self.named_parameters()
                        if "pretrained" not in name
                    ],
                    "lr": self.hparams.lr * 10,  # Decoder learning rate
                },
            ],
            lr=self.hparams.lr,
        )
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=[
                self.hparams.lr,
                self.hparams.lr * 10,
            ],  # Separate max_lr for each param group
            pct_start=0.1,
            # max_lr=self.hparams.lr,
            # pct_start=0.05,
            # cycle_momentum=False,
            # div_factor=1e9,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
