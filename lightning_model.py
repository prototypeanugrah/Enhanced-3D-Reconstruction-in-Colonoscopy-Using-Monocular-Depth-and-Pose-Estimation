"Module for Depth Estimation"

import pytorch_lightning as pl
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor


class DepthEstimationModule(pl.LightningModule):
    def __init__(self, model_name, lr=1e-3):
        super().__init__()
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(
            outputs.predicted_depth,
            targets,
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = torch.nn.functional.mse_loss(
            outputs.predicted_depth,
            targets,
        )
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
