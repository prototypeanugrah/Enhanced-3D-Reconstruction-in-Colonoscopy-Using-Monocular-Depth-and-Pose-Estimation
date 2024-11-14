"Module for inference"

import argparse
import glob
import os

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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
def predict(
    test_dataloader,
    model,
    device,
    op_files,
):

    model.eval()

    for i, sample in tqdm(
        enumerate(test_dataloader),
        desc="Predicting",
        unit="batch",
        total=len(test_dataloader),
    ):
        # Move input to device and get prediction
        sample = sample.to(device)
        pred = model(sample)

        pred = F.interpolate(
            pred[:, None],
            (475, 475),
            mode="bilinear",
            align_corners=True,
        )
        pred = pred.squeeze(1)

        # Process the prediction
        predicted_map = pred.cpu().numpy()
        predicted_map = np.squeeze(predicted_map)  # Remove batch dimension

        # print(f"Pred shape: {pred.shape}")
        # print(f"predicted_map shape: {predicted_map.shape}")

        # exit()

        np.save(
            op_files[i],
            np.float16(predicted_map),
        )


def main(
    checkpoint_path: str = None,
    input_path: str = None,
    model_size: str = None,
):

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    model_encoder = f"vit{model_size}"

    # Load input
    test_vids = [
        os.path.join(input_path, "SyntheticColon_I/Frames_S5/*.png"),
        os.path.join(input_path, "SyntheticColon_I/Frames_S10/*.png"),
        os.path.join(input_path, "SyntheticColon_I/Frames_S15/*.png"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B5/*.png"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B10/*.png"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B15/*.png"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O1/*.png"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O2/*.png"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O3/*.png"),
    ]

    test_rgb = []
    for vid in test_vids:
        if not os.path.exists(os.path.join(os.path.dirname(vid) + "_OP", "depth")):
            os.makedirs(os.path.join(os.path.dirname(vid) + "_OP", "depth"))
        files = sorted(glob.glob(vid))
        test_rgb += [f for f in files if f.split("/")[-1].startswith("Frame")]

    op_files = []
    for filename in test_rgb:
        op_files.append(
            os.path.join(
                os.path.dirname(filename) + "_OP",
                "depth",
                os.path.basename(filename).replace(".png", ".npy"),
            )
        )

    test_dataloader = dataloader.get_dataloaders_test(test_rgb)
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnythingV2(**{**model_configs[model_encoder]}).to(device)
    model.load_state_dict(torch.load(checkpoint_path))

    predict(
        test_dataloader=test_dataloader,
        model=model,
        device=device,
        op_files=op_files,
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
        help="Path to the directory containing the model checkpoint",
    )
    parser.add_argument(
        "-i",
        "--image_path",
        type=str,
        # required=True,
        default="./datasets/",
        help="Path to the directory containing input images",
    )
    parser.add_argument(
        "-m",
        "--model_size",
        type=str,
        default="b",
        choices=["s", "b", "l"],
        help="Size of the DepthAnythingV2 model to use",
    )
    args = parser.parse_args()

    main(
        args.checkpoint_path,
        args.image_path,
        args.model_size,
    )
