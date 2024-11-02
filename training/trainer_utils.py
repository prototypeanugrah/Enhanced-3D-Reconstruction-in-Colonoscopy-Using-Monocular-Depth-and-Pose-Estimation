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

from typing import Dict, List, Union
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
    # BitsAndBytesConfig,
    # Trainer,
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

        # # Quantization configuration (Q-LoRA config)
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name,
            # quantization_config=bnb_config,
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

        # Load the image processor
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # Freeze base model parameters
        for param in self.model.base_model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True

        # Print trainable parameters
        self.print_trainable_parameters()
        self.enable_input_require_grads()

    def forward(
        self,
        x: torch.Tensor = None,
        # pixel_values: torch.Tensor = None,
        # **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input images tensor of shape (B, C, H, W)
            pixel_values: Alternative input format used by Trainer
            **kwargs: Additional keyword arguments

        Returns:
            Model output
        """
        # # Use pixel_values if x is None
        # if x is None:
        #     x = pixel_values

        # Preprocess the input images and move to the device
        x = self.preprocess(x).to(next(self.model.parameters()).device)
        return self.model(x)
    
    def enable_input_require_grads(self):
    """Enable input requires_grad for better gradient flow"""
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    
    self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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

    # def compute_loss(
    #     self,
    #     outputs: torch.Tensor,
    #     targets: torch.Tensor,
    #     max_depth: float = 20.0,
    # ):
    #     """Compute the loss for the Trainer class"""
    #     # predicted_depth = outputs.predicted_depth

    #     # # Resize outputs if needed
    #     # predicted_depth = F.interpolate(
    #     #     predicted_depth.unsqueeze(1),
    #     #     size=targets.shape[-2:],
    #     #     mode="bilinear",
    #     #     align_corners=True,
    #     # ).squeeze(1)

    #     # Create mask for valid depth values
    #     mask = (targets <= max_depth) & (targets >= 0.001)

    #     # Compute MSE loss
    #     loss = nn.MSELoss()(
    #         outputs[mask],
    #         targets[mask],
    #     )

    #     return loss

    def print_trainable_parameters(self):
        """Print detailed information about model parameters"""
        lora_params = 0
        all_params = 0
        for name, param in self.model.named_parameters():
            num_params = param.numel()
            all_params += num_params
            if "lora_" in name:
                lora_params += num_params
            # logger.info(
            #     "LoRA parameter: %s, Shape: %s, Requires grad: %s",
            #     name,
            #     param.shape,
            #     param.requires_grad,
            # )

        logger.info(
            "Total LoRA parameters: %d || Total parameters: %d || LoRA percentage: %.2f%%",
            lora_params,
            all_params,
            100 * lora_params / all_params,
        )


class DepthTrainer(Trainer):
    """
    Trainer class for depth estimation.

    Args:
        train_dataloader (torch.utils.data.DataLoader): Training dataloader.
        eval_dataloader (torch.utils.data.DataLoader): Evaluation dataloader.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        train_dataloader=None,
        eval_dataloader=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.log_dict = {}
        self.training = True
        self.global_step = 0

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Union[List, Dict[str, torch.Tensor]],
        return_outputs: bool = False,
    ) -> torch.Tensor:
        # Handle both list and dictionary input formats
        if isinstance(inputs, (list, tuple)):
            images, targets = inputs
        else:
            images = inputs["pixel_values"]
            targets = inputs["labels"]

        outputs = model(images)

        # Resize outputs to match target size
        outputs = F.interpolate(
            outputs.predicted_depth.unsqueeze(1),  # Add channel dimension
            size=targets.shape[-2:],
            mode="bilinear",
            align_corners=True,
        ).squeeze(
            1
        )  # Remove channel dimension

        loss = model.compute_loss(outputs, targets)

        # Log gradients during training
        if self.training:
            total_norm = 0.0
            # Calculate gradient norms after the backward pass is done by the trainer
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    self.log(
                        {
                            f"gradients/{name}_norm": param_norm.item(),
                        }
                    )
                    total_norm += param_norm.item() ** 2

            total_norm = total_norm**0.5
            self.log(
                {
                    "gradients/total_norm": total_norm,
                }
            )

            # Clear gradients as they will be computed again in the optimizer step
            model.zero_grad()

        mask = (targets <= 20.0) & (
            targets >= 0.001  # Avoid division by zero
        )  # Mask for valid depth values

        # Compute metrics for both training and evaluation
        mask = (targets <= 20.0) & (targets >= 0.001)
        with torch.no_grad():
            metrics = evaluation.compute_errors(
                outputs[mask],
                targets[mask],
            )

            # Update metrics based on training/evaluation mode
            prefix = "eval_" if not self.training else "train_"
            self.log(
                {
                    f"{prefix}loss": loss.item(),
                    f"{prefix}abs_rel": metrics["abs_rel"],
                    f"{prefix}rmse": metrics["rmse"],
                    f"{prefix}delta_1_1": metrics["delta_1_1"],
                }
            )

        return (
            (
                loss,
                outputs,
            )
            if return_outputs
            else loss
        )

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        """Override to use custom dataloader"""
        if self.train_dataloader is None:
            return super().get_train_dataloader()
        return self.train_dataloader

    def get_eval_dataloader(
        self,
        eval_dataset: torch.utils.data.Dataset = None,
    ) -> torch.utils.data.DataLoader:
        """Override to use custom dataloader"""
        if self.eval_dataloader is None:
            return super().get_eval_dataloader(eval_dataset)
        return self.eval_dataloader

    def log_metrics(
        self,
        logs: Dict[str, float],
    ) -> None:
        """Override logging to handle custom metrics"""
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **self.log_dict}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args,
            self.state,
            self.control,
            logs,
        )

    def train(
        self,
        *args,
        **kwargs,
    ):
        """Override train to set training mode"""
        self.training = True
        return super().train(*args, **kwargs)

    def evaluate(
        self,
        *args,
        **kwargs,
    ):
        """Override evaluate to set evaluation mode"""
        self.training = False
        return super().evaluate(*args, **kwargs)

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only=None,
        ignore_keys=None,
    ):
        """Override prediction step to handle our model's output format"""

        # Convert list inputs to dictionary format if needed
        if isinstance(inputs, list):
            inputs = {
                "pixel_values": inputs[0],
                "labels": inputs[1],
            }

        # Get device from model parameters
        device = next(model.parameters()).device

        # Move inputs to the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get model outputs
            outputs = model(inputs["pixel_values"])

            # Resize outputs to match target size if necessary
            predicted_depth = F.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=inputs["labels"].shape[-2:],
                mode="bilinear",
                align_corners=True,
            ).squeeze(1)

            # Compute loss
            loss = model.compute_loss(predicted_depth, inputs["labels"])

            # Return in format expected by trainer
            return (
                loss,
                predicted_depth,  # logits
                inputs["labels"],  # labels
            )

    def evaluation_loop(
        self,
        dataloader,
        description,
        prediction_loss_only=None,
        ignore_keys=None,
        metric_key_prefix="eval",
    ):
        # Get the evaluation loop output from parent class
        eval_output = super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only,
            ignore_keys,
            metric_key_prefix,
        )

        # Ensure eval_loss is in metrics
        if "eval_loss" not in eval_output.metrics:
            eval_output.metrics["eval_loss"] = eval_output.metrics.get(
                f"{metric_key_prefix}_loss", float("inf")
            )

        return eval_output

    def log(
        self,
        logs,
    ):
        """Enhanced logging method that handles tensorboard logging"""
        # Update step counter for training only
        if self.training:
            self.global_step += 1

        # Log to tensorboard if writer is available
        if hasattr(self, "tb_writer") and self.tb_writer:
            for key, value in logs.items():
                self.tb_writer.add_scalar(key, value, self.global_step)

        # Update log dict for parent trainer
        self.log_dict.update(logs)

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        grad_norm=None,  # Add this parameter
    ):
        """Override to handle custom logging"""
        if self.control.should_log:
            logs = {}

            # Add training loss to logs
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["learning_rate"] = self._get_learning_rate()

            # Add all accumulated logs
            logs.update(self.log_dict)
            self.log_dict = {}  # Clear accumulated logs

            self.log(logs)

        return super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
        )