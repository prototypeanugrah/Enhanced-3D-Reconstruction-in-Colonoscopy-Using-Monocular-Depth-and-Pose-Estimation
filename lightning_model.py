"Module for Depth Estimation"

import pytorch_lightning as pl
import torch

from peft import LoraConfig, get_peft_model
from torch.nn import functional
from transformers import (
    AutoModelForDepthEstimation,
    AutoImageProcessor,
    BitsAndBytesConfig,
)


class DepthEstimationModule(pl.LightningModule):
    """
    Module for depth estimation using the DepthAnythingV2 model.

    Args:
        model_name (str): The name of the model to use for depth estimation.
        lr (float): The learning rate for the optimizer. Default is 1e-3.
    """

    def __init__(self, model_name, lr=1e-3):
        super().__init__()

        # Quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            # device_map="auto",
        )
        # print(self.model)
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type="DEPTH_ESTIMATION",
        )
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        # print(self.model)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.lr = lr

        # Freeze all parameters except LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs, targets = (
            batch  # shape: (batch_size, 3, height, width),
            # (batch_size, 3, height, width)
        )
        inputs = inputs.to(torch.float16)
        targets = targets.to(torch.float16)

        outputs = self.model(
            inputs
        ).predicted_depth  # shape: (batch_size, 1, height, width)

        # Ensure outputs have the same spatial dimensions as targets
        outputs = functional.interpolate(
            outputs.unsqueeze(1),  # Add a channel dimension
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            1
        )  # shape: (batch_size, height, width)

        # Ensure targets have only one channel (depth map)
        if targets.shape[1] == 3:
            targets = targets.mean(dim=1, keepdim=True)  # Convert RGB to grayscale
            # shape: (batch_size, 1, height, width)

        # Ensure outputs have the same shape as targets
        outputs = outputs.unsqueeze(1)  # shape: (batch_size, 1, height, width)

        loss = functional.mse_loss(
            outputs,
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

    def validation_step(self, batch):
        inputs, targets = batch
        inputs = inputs.to(torch.float16)
        targets = targets.to(torch.float16)
        outputs = self.model(inputs).predicted_depth
        outputs = functional.interpolate(
            outputs.unsqueeze(1),  # Add a channel dimension
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Ensure targets have only one channel (depth map)
        if targets.shape[1] == 3:
            targets = targets.mean(dim=1, keepdim=True)  # Convert RGB to grayscale
            # shape: (batch_size, 1, resize_height, resize_width)

        # Ensure outputs have the same shape as targets
        outputs = outputs.unsqueeze(1)
        loss = functional.mse_loss(
            outputs,
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
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )
