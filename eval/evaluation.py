"Module for the evaluation metrics"

import torch
import torch.nn.functional as F


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
    # Ensure pred and gt have the same shape
    pred = F.interpolate(
        pred,
        size=gt.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    # Ensure pred has the same number of dimensions as gt
    if pred.dim() == 4 and gt.dim() == 3:
        pred = pred.squeeze(1)

    # # Mask out invalid ground truth values (e.g., zeros)
    # mask = gt > 0
    # pred = pred[mask]
    # gt = gt[mask]

    diff = pred - gt

    abs_rel = torch.mean(torch.abs(diff) / gt)  # Absolute relative error
    sq_rel = torch.mean(torch.pow(diff, 2) / gt) * 1000  # Squared relative error

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))  # Root mean squared error

    # δ<1.1 (percentage of pixels within 10% of actual depth)
    thresh = torch.max(
        (gt / pred),
        (pred / gt),
    )
    delta_1_1 = (thresh < 1.1).float().mean()

    return {
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "delta_1_1": delta_1_1.item(),
    }
