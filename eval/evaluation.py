"Module for the evaluation metrics"

import torch


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

    # * 20 to get centimeters
    diff = (pred - gt) * 20
    epsilon = 1e-6  # Small positive constant

    abs_rel = torch.mean(
        torch.abs(diff) / (gt * 20 + epsilon)
    )  # Absolute relative error
    # sq_rel = (
    #     torch.mean(torch.pow(diff, 2) / (gt + epsilon)) * 1000
    # )  # Squared relative error

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))  # Root mean squared error

    # δ<1.1 (percentage of pixels within 10% of actual depth)
    thresh = torch.max(
        (gt / pred),
        (pred / gt),
    )
    delta_1_1 = (thresh < 1.1).float().mean()

    return {
        "abs_rel": abs_rel.item(),
        # "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "delta_1_1": delta_1_1.item(),
    }
