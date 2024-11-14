"Module for the evaluation metrics"

import logging

import torch

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_errors(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> dict:
    """
    Compute evaluation metrics for depth estimation.

    Args:
        pred (torch.Tensor): Predicted depth map
        gt (torch.Tensor): Ground truth depth map

    Returns:
        dict: Dictionary containing the computed metrics
    """
    assert pred.shape == gt.shape  # Shape 1D tensor

    # Add checks for NaN and Inf values
    if torch.isnan(pred).any():
        logger.warning("NaN values detected in predictions")
    if torch.isinf(pred).any():
        logger.warning("Inf values detected in predictions")

    # * 20 to get centimeters
    diff = pred - gt
    epsilon = 1e-6  # Small positive constant

    L1_error = torch.mean(torch.abs(diff))

    abs_rel = torch.mean(torch.abs(diff) / (gt + epsilon))  # Absolute relative error

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))  # Root mean squared error

    # δ<1.1 (percentage of pixels within 10% of actual depth)
    thresh = torch.max(
        (gt / pred),
        (pred / gt),
    )
    d1 = (thresh < 1.1).float().mean()

    return {
        "d1": d1.detach(),
        "abs_rel": abs_rel.detach(),
        "rmse": rmse.detach(),
        "l1": L1_error.detach(),
    }
