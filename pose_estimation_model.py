"""
This script defines a pose estimation model using PyTorch and PyTorch Lightning.
The model architecture is based on a modified ResNet-18 backbone, with a custom head for predicting 3D poses.
The script includes the following components:

- PoseEstimationNet: A neural network model for pose estimation.
- PoseEstimationModule: A PyTorch Lightning module for training and evaluating the pose estimation model.

The PoseEstimationModule includes methods for:
- Forward pass
- Training step
- Validation step
- Pose loss computation
- Optimizer and learning rate scheduler configuration

Dependencies:
- torch
- torchvision
- torchmetrics
- lightning (PyTorch Lightning)
- eval (custom evaluation module for computing pose errors)
"""

from torchvision.models import resnet18
from torch import optim
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torchmetrics
import lightning as pl

from eval import evaluation


class PoseEstimationNet(nn.Module):
    """
    PoseEstimationNet is a neural network model for pose estimation.
    It uses a modified ResNet-18 backbone with a custom head for predicting 3D poses.

    Args:
        nn (Module): PyTorch neural network module
    """

    def __init__(self, in_channels: int) -> None:
        """
        Initialize the PoseEstimationNet

        Args:
            in_channels (int): Number of input channels (e.g., 8 for
            2 frames x (3 RGB + 1 depth) channels)
        """
        super(PoseEstimationNet, self).__init__()
        # Load a ResNet-18 backbone
        self.backbone = resnet18(weights=None)  # or pretrained if suitable
        # Modify first conv layer to accept in_channels
        self.backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # Replace final FC with a small head that outputs 7D
        num_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_feats, 256)

        # self.pose_head = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 7),  # [tx, ty, tz, qx, qy, qz, qw]
        # )

        self.pose_head = nn.Sequential(
            nn.ReLU(),  # Add ReLU activation
            nn.Dropout(0.3),  # Add dropout layer
            nn.Linear(
                256, 128
            ),  # Add a linear layer with 256 input features and 128 output features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                128, 64
            ),  # Add a linear layer with 128 input features and 64 output features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 7),  # Add a linear layer with 64 input features and
            # 3 output features (tx, ty, tz, qx, qy, qz, qw)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PoseEstimationNet

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Predicted pose tensor of shape (batch_size, 7)
        """
        # x shape: (batch_size, in_channels, H, W)
        features = self.backbone(x)  # shape: (batch_size, 256)
        pose = self.pose_head(features)  # shape: (batch_size, 7)
        return pose


class PoseEstimationModule(pl.LightningModule):
    """
    PoseEstimationModule is a PyTorch Lightning module for training and
    evaluating the pose estimation model.

    Args:
        pl (LightningModule): PyTorch Lightning module
    """

    def __init__(
        self,
        lr: float,
        weight_decay: float,
        pct_start: float,
        div_factor: float,
        cycle_momentum: bool,
        in_channels: int,  # 2 frames × (3 RGB + 1 depth) channels
        beta: float,
        zeta: float,
    ) -> None:
        """
        Initialize the PoseEstimationModule

        Args:
            lr (float): Learning rate for the optimizer
            weight_decay (float): Weight decay for the optimizer
            pct_start (float): Percentage of total steps for the learning rate warm-up
            div_factor (float): Factor by which to divide the initial learning rate
            cycle_momentum (bool): Whether to cycle the momentum during training
            in_channels (int): Number of input channels
            beta: weight from rotation loss
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = PoseEstimationNet(in_channels=in_channels)
        # self.beta = beta
        # self.criterion = nn.MSELoss()

        self.metric = torchmetrics.MetricCollection(
            {
                "ate": torchmetrics.MeanMetric(),  # Absolute translation error
                "rte": torchmetrics.MeanMetric(),  # Relative translation error
                "rote": torchmetrics.MeanMetric(),  # Rotational error
            }
        )

        # Add trajectory buffers
        self.current_trajectory_preds = []
        self.current_trajectory_gts = []
        self.trajectory_metrics = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PoseEstimationModule

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Predicted pose tensor of shape (batch_size, 3)
        """
        return self.model(x)

    def training_step(self, batch: dict) -> torch.Tensor:
        """
        Training step for the PoseEstimationModule.

        Args:
            batch (dict): Dictionary containing the input and target tensors

        Returns:
            torch.Tensor: Loss value for the training step
        """
        input_data = batch["input"]
        target = batch["target"]
        pred = self(input_data)
        loss = self.pose_loss(pred, target)
        self.log("train_loss", loss, prog_bar=True)

        with torch.no_grad():
            # Compute and log evaluation metrics
            metrics = evaluation.compute_pose_errors(
                pred.detach(),
                target.detach(),
            )
            for metric_name, value in metrics.items():
                self.metric[metric_name](value)
                self.log(
                    f"Train/train_{metric_name}",
                    value.item() if torch.is_tensor(value) else value,
                )

        return loss

    def validation_step(self, batch: dict) -> torch.Tensor:
        """
        Validation step for the PoseEstimation

        Args:
            batch (dict): Dictionary containing the input and target tensors

        Returns:
            torch.Tensor: Loss value for the validation step
        """
        input_data = batch["input"]
        target = batch["target"]
        pred = self(input_data)
        loss = self.pose_loss(pred, target)
        self.log("val_loss", loss, prog_bar=True)

        # Store predictions for trajectory evaluation at epoch end
        self.current_trajectory_preds.append(pred.detach().cpu())
        self.current_trajectory_gts.append(target.detach().cpu())

        # Still compute per-frame metrics
        metrics = evaluation.compute_pose_errors(
            pred.detach(),
            target.detach(),
        )
        for metric_name, value in metrics.items():
            self.metric[metric_name](value)
            self.log(
                f"Val/val_{metric_name}",
                value.item() if torch.is_tensor(value) else value,
            )

        return loss

    def on_validation_epoch_start(self):
        # Clear the trajectory buffers at the start of validation
        self.current_trajectory_preds = []
        self.current_trajectory_gts = []

    def on_validation_epoch_end(self) -> None:
        if self.current_trajectory_preds:
            # Find the maximum sequence length
            max_seq_len = max(traj.shape[0] for traj in self.current_trajectory_preds)

            # Pad trajectories to the same length
            padded_preds = []
            padded_gts = []

            for pred_traj, gt_traj in zip(
                self.current_trajectory_preds, self.current_trajectory_gts
            ):
                # Get current sequence length
                seq_len = pred_traj.shape[0]

                # Create padded tensors
                padded_pred = torch.zeros((max_seq_len, 7), device=pred_traj.device)
                padded_gt = torch.zeros((max_seq_len, 7), device=gt_traj.device)

                # Copy the actual data
                padded_pred[:seq_len] = pred_traj
                padded_gt[:seq_len] = gt_traj

                # Add to lists
                padded_preds.append(padded_pred)
                padded_gts.append(padded_gt)

            # Stack the padded trajectories
            pred_trajectory = torch.stack(padded_preds)
            gt_trajectory = torch.stack(padded_gts)

            # Add shape checks
            assert (
                pred_trajectory.shape[-1] == 7
            ), f"Expected 7D poses, got {pred_trajectory.shape[-1]}"
            assert (
                gt_trajectory.shape[-1] == 7
            ), f"Expected 7D poses, got {gt_trajectory.shape[-1]}"

            # Evaluate complete trajectory
            metrics = evaluation.evaluate_trajectory(
                pred_trajectory,
                gt_trajectory,
            )

            # Log metrics
            for metric_name, value in metrics.items():
                self.log(f"Val/val_{metric_name}", value.item())
                self.metric[metric_name](value)

            # Clear the trajectory buffers
            self.current_trajectory_preds = []
            self.current_trajectory_gts = []

    def on_test_epoch_start(self):
        # Reset metrics and trajectory buffers
        self.metric.reset()
        self.current_trajectory_preds = []
        self.current_trajectory_gts = []
        self.trajectory_metrics = []

    def test_step(self, batch: dict, batch_idx: int):
        input_data = batch["input"]
        target = batch["target"]
        pred = self(input_data)

        # Store predictions and ground truth for trajectory evaluation
        self.current_trajectory_preds.append(pred.detach().cpu())
        self.current_trajectory_gts.append(target.detach().cpu())

        # Regular per-frame metrics
        metrics = evaluation.compute_pose_errors(pred.detach(), target.detach())
        for metric_name, value in metrics.items():
            self.metric[metric_name](value)
            self.log(f"Test/test_{metric_name}", value)

        return metrics

    def on_test_epoch_end(self):
        # Convert lists to tensors
        pred_trajectory = torch.stack(self.current_trajectory_preds)
        gt_trajectory = torch.stack(self.current_trajectory_gts)

        # Evaluate complete trajectory
        trajectory_metrics = evaluation.evaluate_trajectory(
            pred_rel_poses=pred_trajectory,
            gt_rel_poses=gt_trajectory,
            initial_pose=None,  # Or provide initial pose if available
        )

        # Log trajectory metrics
        for metric_name, value in trajectory_metrics.items():
            self.log(f"Test/trajectory_{metric_name}", value)

        # Clear buffers
        self.current_trajectory_preds = []
        self.current_trajectory_gts = []

        # Compute and log regular metrics
        final_metrics = self.metric.compute()
        for metric_name, value in final_metrics.items():
            self.log(f"Test/test_{metric_name}", value)

        self.metric.reset()

    def pose_loss(
        self,
        pred_pose: torch.Tensor,
        gt_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the pose loss between predicted and ground truth poses.
        Uses weighted combination of translation L2 loss and quaternion distance.

        Args:
            pred_pose (torch.Tensor): Predicted pose tensor of shape (batch_size, 7)
            gt_pose (torch.Tensor): Ground truth pose tensor of shape (batch_size, 7)

        Returns:
            torch.Tensor: Combined pose loss value
        """
        # pred_pose, gt_pose: (batch_size, 7) -> [tx, ty, tz, qx, qy, qz, qw]
        pred_t = pred_pose[:, :3]
        pred_q = pred_pose[:, 3:]
        gt_t = gt_pose[:, :3]
        gt_q = gt_pose[:, 3:]

        # Add small epsilon to avoid zero norm
        epsilon = 1e-8

        # Add stronger regularization term to penalize quaternions with small norms
        # This encourages the model to predict quaternions with larger norms
        pred_q_norm = pred_q.norm(dim=1, keepdim=True)
        quat_reg_loss = torch.exp(
            -pred_q_norm
        ).mean()  # Higher penalty for smaller norms

        # Normalize both predicted and ground truth quaternions
        pred_q_norm = pred_q.norm(dim=1, keepdim=True).clamp(min=epsilon)
        gt_q_norm = gt_q.norm(dim=1, keepdim=True).clamp(min=epsilon)

        pred_q = pred_q / pred_q_norm
        gt_q = gt_q / gt_q_norm

        # Scale translation loss based on magnitude
        trans_scale = gt_t.norm(dim=1, keepdim=True).clamp(min=epsilon)
        L_t = ((pred_t - gt_t) / trans_scale).pow(2).sum(dim=1).mean()

        # Double cover correction for quaternions
        dot_prod = torch.sum(pred_q * gt_q, dim=1)
        pred_q = torch.where(dot_prod.unsqueeze(1) < 0, -pred_q, pred_q)

        # Quaternion loss with better numerical stability
        L_r = (1 - torch.sum(pred_q * gt_q, dim=1).pow(2)).mean()

        # Add the regularization term to the loss
        # Use a small weight (0.1) to balance with other loss terms
        return L_t + self.hparams.beta * L_r + 0.1 * quat_reg_loss

    def configure_optimizers(self) -> dict:
        """
        Configure the optimizer and scheduler for training.

        Returns:
            dict: Dictionary containing the optimizer and learning rate scheduler
        """
        # Define optimizer (AdamW)
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Define LR scheduler (OneCycleLR)
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=self.hparams.lr,
            pct_start=self.hparams.pct_start,
            div_factor=self.hparams.div_factor,
            cycle_momentum=self.hparams.cycle_momentum,
        )
        # scheduler = lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=0.5,
        #     patience=5,
        #     verbose=True,
        #     min_lr=1e-7,
        # )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                # "interval": "epoch",
                "monitor": "val_loss",
            },
        }
