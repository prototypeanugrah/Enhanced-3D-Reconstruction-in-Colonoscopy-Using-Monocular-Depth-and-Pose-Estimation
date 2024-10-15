import pytorch_lightning as pl
import torch
from transformers import AutoModelForDepthEstimation

class DepthEstimationModule(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # We don't have training data, so this method is not implemented
        pass

    def configure_optimizers(self):
        # We're not training, so we don't need an optimizer
        return None