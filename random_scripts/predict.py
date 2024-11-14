"Module for inference"

import argparse
import os

from tqdm import tqdm
from transformers import AutoModelForDepthEstimation
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# from training import training_utils
from data_processing import dataloader
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from utils import utils


def visualize_results(
    image,
    depth_map,
    gt_depth=None,
    output_path=None,
):
    """
    Visualize the original image, predicted depth map, and ground truth depth
    map (if provided).
    """
    # Convert depth maps to heatmap for visualization
    pred_heatmap = utils.depth_to_heatmap(depth_map)

    if gt_depth is not None:
        # Create figure with four subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
        gt_heatmap = utils.depth_to_heatmap(gt_depth)
    else:
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Plot predicted depth map
    depth_display = ax2.imshow(depth_map, cmap="viridis")
    ax2.set_title("Predicted Depth Map")
    ax2.axis("off")
    plt.colorbar(depth_display, ax=ax2)

    # Plot predicted depth heatmap
    ax3.imshow(pred_heatmap)
    ax3.set_title("Predicted Depth Heatmap")
    ax3.axis("off")

    # Plot ground truth depth map if provided
    if gt_depth is not None:
        ax4.imshow(gt_heatmap)
        ax4.set_title("Ground Truth Depth Map")
        ax4.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    plt.show()


@torch.no_grad()
def test(
    model: DepthAnythingV2,
    dataloader: torch.utils.data.DataLoader,,
) -> tuple:
    """
    Function to test the model on the validation set.

    Args:
        model (DepthAnythingV2): Model to test.
        criterion (nn.MSELoss): Loss function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
        epoch (int): Current epoch.
        writer (SummaryWriter): TensorBoard writer.
        accelerator (Accelerator): Accelerator object.
        max_depth (float): Maximum depth value.

    Returns:
        tuple: Tuple containing the loss and evaluation results.
    """

    # Validation
    model.eval()
    results = {"d1": 0, "abs_rel": 0, "rmse": 0}
    for batch_idx, sample in enumerate(dataloader):

        img, depth = sample

        pred = model(img)
        # evaluate on the original resolution
        pred = F.interpolate(
            pred[:, None],
            depth.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        pred = pred.squeeze(1)

        assert (
            pred.shape == depth.shape
        ), f"Shape mismatch. Target: {depth.shape}, Output: {pred.shape}"

        valid_mask = (depth <= max_depth) & (depth >= 0.001)

        loss = criterion(
            pred[valid_mask],
            depth[valid_mask],
        )
        val_loss += loss.detach()

        # Update progress bar with current loss
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        cur_results = evaluation.compute_errors(
            pred[valid_mask],
            depth[valid_mask],
        )

        if batch_idx % 50 == 0:
            writer.add_scalar(
                "Val/Loss",
                loss.item(),
                epoch * len(dataloader) + batch_idx,
            )

        for k in results.keys():
            results[k] += cur_results[k]

    val_loss /= len(dataloader)
    val_loss = accelerator.reduce(val_loss, reduction="mean").item()

    for k in results.keys():
        results[k] = results[k] / len(dataloader)
        results[k] = round(
            accelerator.reduce(
                results[k],
                reduction="mean",
            ).item(),
            3,
        )
        writer.add_scalar(f"Val/{k}", results[k], epoch)

    return (
        val_loss,
        results,
    )


def main(
    checkpoint_path: str = None,
    input_path: str = None,
    # gt_depth_path: str = None,
    output_path: str = None,
):
    # Load input
    test_vids = [
        os.path.join(input_path, "SyntheticColon_I/Frames_S5"),
        os.path.join(input_path, "SyntheticColon_I/Frames_S10"),
        os.path.join(input_path, "SyntheticColon_I/Frames_S15"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B5"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B10"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B15"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O1"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O2"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O3"),
    ]

    test_rgb = []
    for vid in test_vids:
        _, rgb = utils.load_frames(vid)
        test_rgb.extend(rgb)

    test_dataloader = dataloader.get_dataloaders_test(test_rgb)
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnythingV2(**{**model_configs[model_encoder]}).to(device)
    model.load_state_dict(torch.load(checkpoint_path))

    predict(
        test_dataloader=test_dataloader,
        model=model,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on the trained model",
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        required=True,
        # default="datasets/",
        help="Path to the directory containing the model checkpoint",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        required=True,
        # default="datasets/",
        help="Path to the directory containing input image",
    )
    # parser.add_argument(
    #     "-gt",
    #     "--gt_depth_path",
    #     type=str,
    #     required=True,
    #     # default="datasets/",
    #     help="Path to the directory containing ground truth depth map",
    # )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        # required=True,
        # default="datasets/",
        help="Path to the directory to save the visualization",
    )
    args = parser.parse_args()

    # os.makedirs(args.output_path, exist_ok=True)  # Ensure the output directory exists
    main(
        args.checkpoint_path,
        args.image_path,
        # args.gt_depth_path,
        args.output_path,
    )
