"Module for Depth Estimation Model"

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
        max_depth: float = 20.0,
        lr: float = 5e-6,
        pct_start: float = 0.1,
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

        dataset = "hypersim"
        pretrained_from = (
            f"./base_checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth"
        )
        self.model = dpt.DepthAnythingV2(
            **{
                **self.model_configs[encoder],
                "max_depth": self.hparams.max_depth,
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
        img, depth = batch["image"], batch["depth"]

        # Clamp depth values to min and max values
        depth = torch.clamp(
            depth,
            min=self.hparams.min_depth,
            max=self.hparams.max_depth,
        )
        return img, depth

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
        pred = pred[:, None].clamp(
            self.hparams.min_depth,
            self.hparams.max_depth,
        )

        loss = self.loss(pred, depth)
        self.log(
            "train_loss",
            loss,
            batch_size=img.shape[0],
        )

        # Compute and log evaluation metrics
        metrics = evaluation.compute_errors(
            pred[depth > 1e-4].flatten(),
            depth[depth > 1e-4].flatten(),
        )
        for metric_name, value in metrics.items():
            self.metric[metric_name](value)
            self.log(
                f"Train/train_{metric_name}",
                value,
                batch_size=img.shape[0],
            )

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

        pred = self.model(img)
        pred = pred[:, None].clamp(
            self.hparams.min_depth,
            self.hparams.max_depth,
        )

        loss = self.loss(pred, depth)
        self.log(
            "val_loss",
            loss,
            batch_size=img.shape[0],
        )

        # Compute and log evaluation metrics
        metrics = evaluation.compute_errors(
            pred[depth > 1e-4].flatten(),
            depth[depth > 1e-4].flatten(),
        )
        for metric_name, value in metrics.items():
            self.metric[metric_name](value)
            self.log(
                f"Val/val_{metric_name}",
                value,
                prog_bar=True,
                batch_size=img.shape[0],
            )

        # if batch_idx < 10 and self.logger is not None:
        #     fig = self.trainer.datamodule.val_dataset.plot(
        #         img[0].cpu().detach(),
        #         depth[0].cpu().detach(),
        #         pred[0].cpu().detach(),
        #     )
        #     self.logger.experiment.log({f"val_{batch_idx}": fig})
        #     plt.close(fig)

        return loss

    def on_test_epoch_start(self):
        # Reset all metrics at the start of the test epoch
        self.metric.reset()

    def test_step(
        self,
        batch: dict,
    ) -> None:
        """
        Perform a test step.

        Args:
            batch (dict): A dict containing the input image and the target
        """
        img, depth = self._preprocess_batch(batch)

        pred = self.model(img)
        pred = pred[:, None].clamp(
            self.hparams.min_depth,
            self.hparams.max_depth,
        )

        # Compute and log evaluation metrics
        metrics = evaluation.compute_errors(
            pred[depth > 1e-4].flatten(),
            depth[depth > 1e-4].flatten(),
        )

        for metric_name, value in metrics.items():
            self.metric[metric_name](value)

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

        pred = self.model(img)
        pred = pred[:, None].clamp(
            self.hparams.min_depth,
            self.hparams.max_depth,
        )

        return pred

    def configure_optimizers(self) -> dict:
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
                    "lr": self.hparams.lr * 2,  # Decoder learning rate
                },
            ]
        )
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=self.hparams.lr * 2,
            # max_lr=[
            #     self.hparams.lr,
            #     self.hparams.lr * 2,
            # ],
            pct_start=self.hparams.pct_start,
            # max_lr=self.hparams.lr,
            # pct_start=0.05,
            # cycle_momentum=False,
            # div_factor=1e9,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
