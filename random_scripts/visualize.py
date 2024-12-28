import argparse
import os

from torchvision import transforms
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data_processing import dataloader
from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2
from utils import utils


def visualize_results(
    image,
    depth_map,
    gt_depth=None,
    output_path=None,
):
    """
    Visualize the original image, predicted depth map, and ground truth depth map.
    """
    # Ensure everything is on CPU and in numpy format
    if torch.is_tensor(image):
        image = image.cpu().numpy()
        # Convert from (B, C, H, W) or (C, H, W) to (H, W, C)
        if len(image.shape) == 4:
            image = image.squeeze(0)  # Remove batch dimension
        image = image.transpose(1, 2, 0)  # Change from CHW to HWC

    if torch.is_tensor(depth_map):
        depth_map = depth_map.cpu().numpy()
    if torch.is_tensor(gt_depth):
        gt_depth = gt_depth.cpu().numpy()

    # Create figure
    if gt_depth is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot original image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    # Plot predicted depth map
    depth_display = ax2.imshow(depth_map, cmap="Spectral_r")
    ax2.set_title("Predicted Depth Map")
    ax2.axis("off")
    plt.colorbar(depth_display, ax=ax2)

    # Plot ground truth if provided
    if gt_depth is not None:
        ax3.imshow(gt_depth, cmap="Spectral_r")
        ax3.set_title("Ground Truth Depth Map")
        ax3.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    plt.show()


def main(
    checkpoint_path: str = None,
    input_path: str = None,
    output_path: str = None,
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
        # os.path.join(input_path, "SyntheticColon_I/Frames_S1"),
        # os.path.join(input_path, "SyntheticColon_I/Frames_S10"),
        # os.path.join(input_path, "SyntheticColon_I/Frames_S15"),
        # os.path.join(input_path, "SyntheticColon_II/Frames_B5"),
        # os.path.join(input_path, "SyntheticColon_II/Frames_B10"),
        # os.path.join(input_path, "SyntheticColon_II/Frames_B15"),
        # os.path.join(input_path, "SyntheticColon_III/Frames_O1"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O2"),
        # os.path.join(input_path, "SyntheticColon_III/Frames_O3"),
    ]

    test_rgb = []
    test_depth = []
    for vid in test_vids:
        depth, rgb = utils.load_frames(vid)
        test_rgb.extend(rgb)
        test_depth.extend(depth)

    # test_dataloader = dataloader.get_dataloaders_test(test_rgb)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthAnythingV2(
        **{
            **model_configs[model_encoder],
            "max_depth": 20.0,
        }
    )

    # Load checkpoint and fix state dict keys
    checkpoint = torch.load(checkpoint_path)
    if "state_dict" in checkpoint:
        print("Getting state dict from checkpoint['state_dict']")
        state_dict = checkpoint["state_dict"]

        # Fix the key prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("model."):
                # Remove the "model." prefix
                new_key = key[6:]  # Skip first 6 characters ("model.")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        state_dict = new_state_dict

    model.load_state_dict(
        state_dict,
        # strict=False,
    )  # Use strict=False to ignore missing keys

    # model.load_state_dict(torch.load(checkpoint_path))

    model = model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (518, 518),
                # interpolation=cv2.INTER_CUBIC,
                antialias=True,
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    for idx, (image_path, depth_path) in enumerate(zip(test_rgb, test_depth)):
        # Load and preprocess image_path
        image_path = cv2.imread(image_path)
        if image_path is None:
            print(f"Failed to load image_path at index {idx}")
            continue

        # Convert BGR to RGB
        image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB) / 255.0
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        image = transform(image)
        image = image.float()

        # Add batch dimension if not present
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # Move to device
        image = image.to(device)

        # Get prediction
        with torch.no_grad():
            pred = model(image)
            pred = F.interpolate(
                pred[:, None],
                depth.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )

            pred = pred.squeeze().cpu().numpy()

            assert (
                pred.shape == depth.shape
            ), f"Shape mismatch. Pred shape: {pred.shape}, GT shape: {depth.shape}"

            pred = (pred - pred.min()) / (pred.max() - pred.min()) * 255.0
            pred = pred.astype(np.uint8)

            cmap = matplotlib.colormaps.get_cmap("Spectral")

            pred = (cmap(pred)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            # Visualize results
            output_file = (
                os.path.join(output_path, f"visualization_{idx}.png")
                if output_path
                else None
            )

            visualize_results(
                image=image,
                depth_map=pred,
                gt_depth=depth,
                output_path=output_file,
            )

        if idx >= 5:
            break

    # # Select 5 random indices
    # random_indices = np.random.choice(len(test_rgb), 5, replace=False)

    # # Create the transform for preprocessing
    # transform_input = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Resize(
    #             (350, 350),
    #             antialias=True,
    #         ),
    #         transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225],
    #         ),
    #     ]
    # )

    # # Get predictions for the selected frames
    # with torch.no_grad():
    #     for idx in random_indices:
    #         # Get the input image and corresponding ground truth
    #         image_path = test_rgb[idx]
    #         depth_path = test_depth[idx]

    #         # Load and convert image using PIL
    #         # gt_depth = np.array(Image.open(depth_path))  # Load depth map

    #         image = cv2.imread(image_path)
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         # y = np.array(Image.open(target_id)) / 255 / 256
    #         gt_depth = cv2.imread(
    #             depth_path, cv2.IMREAD_UNCHANGED
    #         )  # Read as-is without any conversion
    #         gt_depth = (
    #             gt_depth.astype(np.float32) / 255.0 / 256.0
    #         )  # Convert to float and normalize

    #         # Prepare input for model (assuming your dataloader handles the preprocessing)
    #         inputs = transform_input(image).unsqueeze(0).to(device)

    #         h, w = gt_depth.shape[-2:]

    #         # Get prediction
    #         outputs = model(inputs)
    #         pred = F.interpolate(
    #             outputs[:, None],
    #             (h, w),
    #             mode="bilinear",
    #             align_corners=True,
    #         )
    #         pred = pred.squeeze(1)
    #         pred = pred.squeeze()

    #         assert (
    #             pred.shape == gt_depth.shape
    #         ), f"Shape mismatch. Pred shape: {pred.shape}, GT shape: {gt_depth.shape}"

    #         pred_depth = Image.fromarray(pred.cpu().numpy().astype("uint16"))

    #         # Visualize results
    #         output_file = (
    #             os.path.join(output_path, f"visualization_{idx}.png")
    #             if output_path
    #             else None
    #         )
    #         visualize_results(
    #             image=image,
    #             depth_map=pred_depth,
    #             gt_depth=gt_depth,
    #             output_path=output_file,
    #         )


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
        "--input_path",
        type=str,
        # required=True,
        default="./datasets/SyntheticColon/",
        help="Path to the directory containing input images",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        help="Path to the directory to save the visualization",
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

    os.makedirs(args.output_path, exist_ok=True)  # Ensure the output directory exists
    main(
        args.checkpoint_path,
        args.input_path,
        args.output_path,
        args.model_size,
    )
