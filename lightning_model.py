"Module for Depth Estimation Model Config"

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from transformers import (
    AutoModelForDepthEstimation,
    AutoImageProcessor,
    BitsAndBytesConfig,
)

import evaluation


class DepthEstimationModule(pl.LightningModule):
    """
    Module for depth estimation using the DepthAnythingV2 model.

    Args:
        model_name (str): The name of the model to use for depth estimation.
        lr (float): The learning rate for the optimizer. Default is 1e-3.
    """

    def __init__(
        self,
        model_name,
        lr=1e-3,
        use_scheduler=False,
        warmup_steps=1000,
    ):
        """
        Initialize the DepthEstimationModule.
        """
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
        # LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
            task_type="DEPTH_ESTIMATION",
        )
        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.lr = lr
        self.use_scheduler = use_scheduler
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()

        # Freeze all parameters except LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        """
        Perform a single training step.

        Args:
            batch (tuple): A tuple containing input images and target depth maps.

        Returns:
            torch.Tensor: The computed loss for this training step.
        """
        inputs, targets = (
            batch  # shape: (batch_size, 3, height, width),
            # (batch_size, 3, height, width)
        )
        inputs = inputs.to(torch.float16)
        targets = targets.to(torch.float16)

        # Extract the predicted_depth from the outputs
        outputs = self.model(
            inputs
        ).predicted_depth  # shape: (batch_size, 1, height, width)

        # Ensure outputs have the same spatial dimensions as targets
        outputs = F.interpolate(
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

        loss = nn.MSELoss()(
            outputs,
            targets,
        )

        # Check if gradients are being computed
        if self.global_step % 100 == 0:  # Check every 100 steps
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if param.grad is None:
                        print(f"No gradient for {name}")
                    elif param.grad.abs().sum().item() == 0:
                        print(f"Zero gradient for {name}")

        # Log the training loss
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Compute and log additional metrics
        metrics = evaluation.compute_errors(outputs, targets)
        for name, value in metrics.items():
            self.log(
                f"train_{name}",
                value,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        return loss

    def validation_step(self, batch):
        """
        Perform a single validation step.

        Args:
            batch (tuple): A tuple containing input images and target depth maps.

        Returns:
            torch.Tensor: The computed loss for this validation step.
        """
        inputs, targets = batch
        inputs = inputs.to(torch.float16)
        targets = targets.to(torch.float16)
        outputs = self.model(inputs).predicted_depth
        outputs = F.interpolate(
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

        loss = nn.MSELoss()(
            outputs,
            targets,
        )
        # Log the validation loss for each step
        self.log(
            "val_loss_step",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        # Compute metrics
        metrics = evaluation.compute_errors(outputs, targets)

        # Log metrics
        for name, value in metrics.items():
            self.log(
                f"val_{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )

        return {"val_loss": loss, **metrics}

    def lr_scheduler_step(self, scheduler, metric):
        """
        Step the learning rate scheduler based on the validation metric.
        """
        scheduler.step(metric)

    def on_validation_epoch_end(self):
        """Log the learning rate at the end of each validation epoch"""
        optimizer = self.optimizers()
        # metrics_mean = {}

        # for metric in self.validation_step_outputs[0].keys():
        #     metrics_mean[metric] = torch.mean(
        #         torch.tensor([x[metric] for x in self.validation_step_outputs])
        #     )
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=True)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Log the learning rate at the end of each training step"""
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        current_lr = optimizer.param_groups[0]["lr"]
        self.log("learning_rate", current_lr, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            dict: The optimizer and learning rate scheduler.
        """
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
        )
        if self.use_scheduler:
            scheduler = WarmupReduceLROnPlateau(
                optimizer,
                warmup_steps=self.warmup_steps,
                factor=0.5,
                patience=5,
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            return optimizer


class WarmupReduceLROnPlateau:
    """
    A learning rate scheduler that combines warmup and ReduceLROnPlateau.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        factor=0.1,
        patience=5,
        verbose=True,
    ):
        """
        Initialize the WarmupReduceLROnPlateau scheduler.

        Args:
            optimizer (torch.optim.Optimizer): The optimizer to adjust the
            learning rate for.
            warmup_steps (int): The number of warmup steps to linearly increase
            the learning rate.
            factor (float, optional): The factor to reduce the learning rate
            by. Defaults to 0.1.
            patience (int, optional): The number of epochs to wait before
            reducing the learning rate. Defaults to 5.
            verbose (bool, optional): Whether to print updates to the console.
            Defaults to True.
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.patience = patience
        self.verbose = verbose
        self.step_count = 0
        self.best = float("inf")
        self.num_bad_epochs = 0
        self.mode = "min"
        self.reduce_lr_on_plateau = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            verbose=verbose,
        )
        # Store initial learning rates
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, metrics):
        """
        Step the learning rate scheduler.

        Args:
            metrics (float): The validation loss to use for the learning rate
        """
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # During warmup, linearly increase learning rate
            progress = self.step_count / self.warmup_steps
            for param_group, initial_lr in zip(
                self.optimizer.param_groups, self.initial_lrs
            ):
                param_group["lr"] = progress * initial_lr
        else:
            # After warmup, use ReduceLROnPlateau
            self.reduce_lr_on_plateau.step(metrics)

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "reduce_lr_on_plateau": self.reduce_lr_on_plateau.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.step_count = state_dict["step_count"]
        self.best = state_dict["best"]
        self.num_bad_epochs = state_dict["num_bad_epochs"]
        self.reduce_lr_on_plateau.load_state_dict(
            state_dict["reduce_lr_on_plateau"],
        )
