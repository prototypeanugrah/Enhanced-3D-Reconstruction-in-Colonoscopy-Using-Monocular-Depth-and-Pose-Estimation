"""
Module for training utilities.
Includes:
- DepthEstimationModule: Module for depth estimation using the DepthAnythingV2
model.
- WarmupReduceLROnPlateau: A learning rate scheduler that combines warmup and
ReduceLROnPlateau.
- EarlyStopping: Early stops the training if validation loss doesn't improve
after a given patience.
- test: Function for testing the model.
- train: Function for training the model.
- train_step: Function for a single training step.
- validate_step: Function for a single validation step.
"""

import logging

from peft import LoraConfig, get_peft_model
from torch import nn
from torch import amp
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AutoModelForDepthEstimation,
    AutoImageProcessor,
    BitsAndBytesConfig,
)
import torch
import torch.nn.functional as F

from eval import evaluation

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DepthEstimationModule(nn.Module):
    """
    Module for depth estimation using the DepthAnythingV2 model.

    Args:
        model_name (str): The name of the model to use for depth estimation.
        lora_r (int, optional): The LoRA rank value to use. Defaults to 4.
        device (str, optional): The device to use for training. Defaults to "cuda".
    """

    def __init__(
        self,
        model_name: str,
        lora_r: int = 4,
        device: str = "cuda",
    ):
        super().__init__()

        # Quantization configuration (Q-LoRA config)
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
        )

        # LoRA configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=32,
            target_modules=[
                "query",
                "value",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="DEPTH_ESTIMATION",
            init_lora_weights=True,
        )

        # Apply LoRA to the model
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(device)

        # Print trainable parameters
        self.print_trainable_parameters()

        # Load the image processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Freeze all parameters except LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input images tensor of shape (B, C, H, W)

        Returns:
            Model output
        """
        # Preprocess the input images and move to the device
        x = self.preprocess(x).to(next(self.model.parameters()).device)
        return self.model(x)

    def preprocess(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Preprocess images using the model's processor.

        Args:
            images: Input images tensor of shape (B, C, H, W)

        Returns:
            Preprocessed images tensor
        """
        # Normalize the input tensor to [0, 1] range before processing
        images = (images - images.min()) / (images.max() - images.min() + 1e-8)

        # Process images using the processor
        processed = self.processor(
            images=images,
            return_tensors="pt",
            do_rescale=False,
        )
        return processed.pixel_values

    def print_trainable_parameters(self):
        """Helper method to print trainable parameter info"""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        logger.info(
            "Trainable params: %d || all params: %d || trainable (perc) %.2f",
            trainable_params,
            all_param,
            100 * trainable_params / all_param,
        )


class WarmupReduceLROnPlateau:
    """
    A learning rate scheduler that combines warmup and ReduceLROnPlateau.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 500,
        factor: float = 0.1,
        patience: int = 5,
        verbose: bool = True,
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

    def step(
        self,
        metrics: float,
    ) -> None:
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

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "best": self.best,
            "num_bad_epochs": self.num_bad_epochs,
            "reduce_lr_on_plateau": self.reduce_lr_on_plateau.state_dict(),
        }

    def load_state_dict(
        self,
        state_dict: dict,
    ) -> None:
        self.step_count = state_dict["step_count"]
        self.best = state_dict["best"]
        self.num_bad_epochs = state_dict["num_bad_epochs"]
        self.reduce_lr_on_plateau.load_state_dict(
            state_dict["reduce_lr_on_plateau"],
        )


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given
    patience.
    """

    def __init__(
        self,
        patience: int = 5,
        verbose: bool = False,
        delta: float = 1e-5,
        path: str = "checkpoint.pt",
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss
            improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss
            improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify
            as an improvement. Default: 0
            path (str): Path for the checkpoint to be saved to.
            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.path = path

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
    ):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                "EarlyStopping counter: %d out of %d",
                self.counter,
                self.patience,
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(
        self,
        val_loss: float,
        model: nn.Module,
    ) -> None:
        """
        Saves model when validation loss decrease.

        Args:
            val_loss (float): Validation loss to determine if the model has improved.
            model (nn.Module): Model to save.
        """
        if self.verbose:
            logger.info(
                "Validation loss decreased (%.6f --> %.6f). Saving model ...",
                self.val_loss_min,
                val_loss,
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    epoch: int,
    device: torch.device,
    writer: SummaryWriter,
) -> float:
    """
    Validate the model on the validation set.

    Args:
        model (nn.Module): Model to validate.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        epoch (int): Current epoch.
        device (torch.device): Device to use for validation.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer.

    Returns:
        float: Validation loss.
    """

    model.eval()

    val_loss = 0.0
    val_metrics = {}

    with tqdm(
        val_loader,
        desc=f"Validation Epoch {epoch+1}",
        leave=False,
    ) as pbar:
        for batch in pbar:
            loss, metrics = validate_step(
                model,
                batch,
                device,
            )
            val_loss += loss.item()
            for k, v in metrics.items():
                val_metrics[k] = val_metrics.get(k, 0) + v
            pbar.set_postfix({"loss": loss.item()})

    val_loss /= len(val_loader)

    val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}

    # Log validation metrics
    writer.add_scalar(
        "Val/Loss",
        val_loss,
        epoch,
    )
    for k, v in val_metrics.items():
        writer.add_scalar(
            f"Val/{k}",
            v,
            epoch,
        )

    return val_loss


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: torch.utils.tensorboard.SummaryWriter,
    draft: bool = False,
) -> float:
    """
    Train the model on the validation set.

    Args:
        model (nn.Module): Model to train.
        val_loader (torch.utils.data.DataLoader): Train data loader.
        epoch (int): Current epoch.
        device (torch.device): Device to use for train.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer.

    Returns:
        float: Train loss.
    """

    model.train()

    scaler = amp.GradScaler()
    train_loss = 0.0
    train_metrics = {}

    with tqdm(
        train_loader,
        desc=f"Training Epoch {epoch+1}",
        leave=False,
    ) as pbar:
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            with amp.autocast(device_type=device.type):
                loss, metrics = train_step(
                    model,
                    batch,
                    device,
                )
            scaler.scale(loss).backward()

            # Scaled gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            for k, v in metrics.items():
                train_metrics[k] = train_metrics.get(k, 0) + v

            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})

            # Log training metrics
            if batch_idx % 100 == 0:
                writer.add_scalar(
                    "Train/Loss",
                    loss.item(),
                    epoch * len(train_loader) + batch_idx,
                )

    train_loss /= len(train_loader)

    # Log the error metrics
    train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}
    for k, v in train_metrics.items():
        writer.add_scalar(
            f"Train/{k}",
            v,
            epoch,
        )
    return train_loss


def train_step(
    model: nn.Module,
    batch: tuple,
    device: torch.device,
) -> tuple:
    """
    Perform a single training step.

    Args:
        model (nn.Module): Model to train.
        batch (tuple): A tuple containing the input images and target depth
        device (torch.device): Device to use for training.

    Returns:
        tuple: A tuple containing the loss and metrics.
    """

    # The last layer in our model is a sigmoid for each pixel, producing an
    # output from 0 to 1. We simply multiply each pixel by "max_depth" to
    # represent distances from 0 to "max_depth".
    max_depth = 20.0
    inputs, targets = batch  # Shape: (batch_size, 3, H, W), (batch_size, H, W)

    # Move the inputs and targets to the device and cast to float16
    inputs = inputs.to(device, dtype=torch.float16)
    targets = targets.to(device, dtype=torch.float16)

    outputs = model(inputs).predicted_depth  # Shape: (batch_size, H, W)

    # Add check for NaN/Inf values
    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        logger.warning("Model produced NaN or Inf values in outputs")

    # Resize the output to match the target size if necessary
    outputs = (
        F.interpolate(
            outputs.unsqueeze(1),
            size=targets.shape[-2:],  # Match the target size
            mode="bilinear",
            align_corners=True,
        )
        .squeeze(1)
        .to(dtype=torch.float16)
    )  # Shape: (batch_size, H, W)

    assert outputs.shape == targets.shape, "Training: Output and target shape mismatch", f'Output shape: {outputs.shape}', f'Target shape: {targets.shape}'

    mask = (targets <= max_depth) & (
        targets >= 0.001 # Avoid division by zero
    )  # Mask for valid depth values

    # Compute the MSE loss
    loss = nn.MSELoss()(
        outputs[mask],
        targets[mask],
    )

    # Compute metrics
    metrics = evaluation.compute_errors(
        outputs[mask],
        targets[mask],
    )

    # Force backward pass through LoRA layers
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n]
    grad_tensors = [torch.ones_like(p) for p in lora_params]
    torch.autograd.backward(
        lora_params,
        grad_tensors,
        retain_graph=True,
    )

    return (
        loss,
        metrics,
    )


def validate_step(
    model: nn.Module,
    batch: tuple,
    device: torch.device,
) -> tuple:
    """
    Perform a single validation step.

    Args:
        model (nn.Module): Model to validate.
        batch (tuple): A tuple containing the input images and target depth
        device (torch.device): Device to use for validating.

    Returns:
        tuple: A tuple containing the loss and metrics.
    """

    # The last layer in our model is a sigmoid for each pixel, producing an
    # output from 0 to 1. We simply multiply each pixel by "max_depth" to
    # represent distances from 0 to "max_depth".
    max_depth = 20.0
    inputs, targets = batch

    # Move the inputs and targets to the device and cast to float16
    inputs = inputs.to(device, dtype=torch.float16)
    targets = targets.to(device, dtype=torch.float16)

    with torch.no_grad():
        outputs = model(inputs).predicted_depth

        # Resize the output to match the target size if necessary
        outputs = (
            F.interpolate(
                outputs.unsqueeze(1),
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
            .squeeze(1)
            .to(dtype=torch.float16)
        )

        assert (
            outputs.shape == targets.shape
        ), "Validation: Output and target shape mismatch"

        mask = (targets <= max_depth) & (
            targets >= 0.001  # Avoid division by zero
        )  # Mask for valid depth values

        loss = nn.MSELoss()(
            outputs[mask],
            targets[mask],
        )
        metrics = evaluation.compute_errors(
            outputs[mask],
            targets[mask],
        )

    return (
        loss,
        metrics,
    )
