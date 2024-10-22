"Module for the evaluation metrics"

import torch
import torch.nn.functional as F


def compute_errors(pred, gt):
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

    # Mask out invalid ground truth values (e.g., zeros)
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    # thresh = torch.max((gt / pred), (pred / gt))  # Threshold for accuracy metrics
    # a1 = (thresh < 1.25).float().mean()
    # a2 = (thresh < 1.25**2).float().mean()
    # a3 = (thresh < 1.25**3).float().mean()

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)  # Absolute relative error
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)  # Squared relative error

    rmse = torch.sqrt(torch.mean((gt - pred) ** 2))  # Root mean squared error
    rmse_log = torch.sqrt(
        torch.mean((torch.log(gt) - torch.log(pred)) ** 2)
    )  # Root mean squared logarithmic error

    # log10 = torch.mean(torch.abs(torch.log10(gt) - torch.log10(pred)))

    return {
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        # "a1": a1.item(),
        # "a2": a2.item(),
        # "a3": a3.item(),
        # "log10": log10.item(),
    }
