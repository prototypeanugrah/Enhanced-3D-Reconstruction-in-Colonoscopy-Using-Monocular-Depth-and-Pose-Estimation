"Module for inference"

import argparse
import os

from tqdm import tqdm
from transformers import AutoModelForDepthEstimation
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from data_processing import dataloader

# from training import training_utils
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
def predict(
    test_dataloader,
    model,
    device,
):

    model.eval()

    for data in tqdm(
        test_dataloader,
        desc="Predicting",
        unit="batch",
    ):
        # Move input to device and get prediction
        pixel_values = data.to(device)
        outputs = model(pixel_values)
        predicted_depth = outputs.predicted_depth

        # Process the prediction
        predicted_map = predicted_depth.cpu().numpy()
        predicted_map = np.squeeze(predicted_map)  # Remove batch dimension

        # Save raw depth predictions as numpy array
        os.makedirs(f"./Predictions/numpy", exist_ok=True)
        np.save(
            f"./Predictions/numpy/{i:04d}.npy",
            predicted_map,
        )

        # Save visualization as PNG
        os.makedirs(f"./Predictions/png", exist_ok=True)
        predicted_map_viz = (predicted_map * 255 * 256).astype(
            "uint16"
        )  # Scale for visualization
        cv2.imwrite(
            f"./Predictions/png/{i:04d}.png",
            predicted_map_viz,
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
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint_path)
    model = model.to(device)

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

    os.makedirs(args.output_path, exist_ok=True)  # Ensure the output directory exists
    main(
        args.checkpoint_path,
        args.image_path,
        # args.gt_depth_path,
        args.output_path,
    )
