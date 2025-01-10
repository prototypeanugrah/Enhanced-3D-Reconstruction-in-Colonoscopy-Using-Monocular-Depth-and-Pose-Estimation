"""
Born out of Depth Anything V1 Issue 36
Make sure you have the necessary libraries installed.
Code by @1ssb

This script processes a set of images to generate depth maps and corresponding point clouds.
The resulting point clouds are saved in the specified output directory.

Usage:
    python script.py --encoder vitl --load-from path_to_model --max-depth 20 --img-path path_to_images --outdir output_directory --cam-file camera_intrinsics_file

Arguments:
    --encoder: Model encoder to use. Choices are ['vits', 'vitb', 'vitl', 'vitg'].
    --load-from: Path to the pre-trained model weights.
    --max-depth: Maximum depth value for the depth map.
    --img-path: Path to the input image or directory containing images.
    --outdir: Directory to save the output point clouds.
    --cam-file: camera intrinsics file
"""

import argparse
import cv2
import glob
import numpy as np
import open3d as o3d
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import torch

from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2


def read_cam_file(cam_file: str) -> dict:
    with open(cam_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        lines = list(map(float, lines[0].split()))
        fx = lines[0]
        fy = lines[4]
        cx = lines[2]
        cy = lines[5]
        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
        }


def load_transformation(
    position_file,
    rotation_file,
):
    """Loads position and rotation quaternion from text files and converts to
    a 4x4 transformation matrix."""
    position = np.loadtxt(position_file)
    quaternion = np.loadtxt(rotation_file)

    # Convert quaternion to rotation matrix using scipy
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    # Create transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = position
    return transformation


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate depth maps and point clouds from images."
    )
    parser.add_argument(
        "--encoder",
        default="vitl",
        type=str,
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Model encoder to use.",
    )
    parser.add_argument(
        "--load-from",
        default="",
        type=str,
        required=True,
        help="Path to the pre-trained model weights.",
    )
    parser.add_argument(
        "--max-depth",
        default=20,
        type=float,
        help="Maximum depth value for the depth map.",
    )
    parser.add_argument(
        "--img-path",
        type=str,
        required=True,
        help="Path to the input image or directory containing images.",
    )
    # parser.add_argument(
    #     "--outdir",
    #     type=str,
    #     default="./vis_pointcloud",
    #     help="Directory to save the output point clouds.",
    # )
    parser.add_argument(
        "--cam-file",
        type=str,
        # required=True,
        help="Path to the camera intrinsics file",
    )
    parser.add_argument("-o", "--outdir", type=str)
    parser.add_argument("-d", "--ds_type", type=str)
    # parser.add_argument(
    #     "--focal-length-x",
    #     default=470.4,
    #     type=float,
    #     help="Focal length along the x-axis.",
    # )
    # parser.add_argument(
    #     "--focal-length-y",
    #     default=470.4,
    #     type=float,
    #     help="Focal length along the y-axis.",
    # )

    args = parser.parse_args()

    # Determine the device to use (CUDA, MPS, or CPU)
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Model configuration based on the chosen encoder
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
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

    # Initialize the DepthAnythingV2 model with the specified configuration
    depth_anything = DepthAnythingV2(
        **{
            **model_configs[args.encoder],
            "max_depth": args.max_depth,
        }
    )
    # Load checkpoint and fix state dict keys
    checkpoint = torch.load(args.load_from, map_location="cpu")
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

        depth_anything.load_state_dict(state_dict)
    else:
        depth_anything.load_state_dict(checkpoint)

    depth_anything = depth_anything.to(device).eval()

    # Get the list of image files to process
    filenames = []
    if os.path.isfile(args.img_path):
        if args.img_path.endswith("txt"):
            with open(args.img_path, "r", encoding="utf-8") as f:
                filenames = f.read().splitlines()
        else:
            # Single image processing
            filenames = [args.img_path]
            if args.outdir is None:
                args.outdir = str(Path(args.img_path).parent)
    elif args.ds_type == "simcol":
        # SimCol dataset processing
        base_dir = Path(args.img_path)
        for suffix in ["I", "II", "III"]:
            pattern = f"SyntheticColon_{suffix}/Frames_*/FrameBuffer_*.png"
            # filenames.extend(
            #     glob.glob(
            #         str(base_dir / pattern),
            #         recursive=True,
            #     )
            # )
            paths = [
                p
                for p in glob.glob(
                    str(base_dir / pattern),
                    recursive=True,
                )
                if "_OP" not in str(p)
            ]
            filenames.extend(paths)
        if args.outdir is None:
            args.outdir = str(base_dir)
            # frame_path = Path(filenames[0]).parent
            # # print(f"Frame path: {frame_path}")
            # args.outdir = str(frame_path.parent / (frame_path.name + "_PC"))
            # # print(f"Outdir: {args.outdir}")

    elif args.ds_type == "testing":
        base_dir = Path(args.img_path)
        pattern = "frame_*.jpg"
        filenames.extend(
            glob.glob(
                str(base_dir / pattern),
                recursive=True,
            )
        )
        if args.outdir is None:
            args.outdir = str(base_dir)

    # Create the output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    print(f"outdir: {args.outdir}")

    # Process each image file
    # for k, filename in enumerate(filenames):
    for filename in tqdm(
        filenames,
        desc="Processing images",
        unit="image",
    ):

        # For SimCol dataset, determine the correct cam.txt file based on the suffix
        if args.ds_type == "simcol":
            file_path = Path(filename)
            base_dir = Path("datasets/SyntheticColon")  # Set the correct base directory
            if "SyntheticColon_I" in str(file_path):
                cam_file = base_dir / "SyntheticColon_I/cam.txt"
            elif "SyntheticColon_II" in str(file_path):
                cam_file = base_dir / "SyntheticColon_II/cam.txt"
            elif "SyntheticColon_III" in str(file_path):
                cam_file = base_dir / "SyntheticColon_III/cam.txt"
            else:
                raise ValueError(f"Unknown SyntheticColon suffix in {filename}")
        elif args.cam_file:
            cam_file = args.cam_file
        else:
            raise ValueError("No camera file specified. Use --cam-file.")

        # Read camera intrinsics for current image
        cam_intrinsics = read_cam_file(str(cam_file))
        fx, fy, cx, cy = (
            cam_intrinsics["fx"],
            cam_intrinsics["fy"],
            cam_intrinsics["cx"],
            cam_intrinsics["cy"],
        )

        # Get the output directory specific to this frame's sequence
        if args.ds_type == "simcol":
            frame_path = Path(filename).parent
            outdir = str(frame_path.parent / (frame_path.name + "_PC"))
            os.makedirs(outdir, exist_ok=True)
        else:
            outdir = args.outdir

        # Load the image
        color_image = Image.open(filename).convert("RGB")
        width, height = color_image.size

        # Read the image using OpenCV
        image = cv2.imread(filename)
        pred = depth_anything.infer_image(image, height)

        # Resize depth prediction to match the original image size
        resized_pred = Image.fromarray(pred).resize(
            (width, height),
            Image.NEAREST,
        )

        # Generate mesh grid and calculate point cloud coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        # x = (x - width / 2) / args.focal_length_x
        # y = (y - height / 2) / args.focal_length_y
        x = (x - cx) / fx
        y = (y - cy) / fy
        z = np.array(resized_pred)
        points = np.stack(
            (
                np.multiply(x, z),
                np.multiply(y, z),
                z,
            ),
            axis=-1,
        ).reshape(-1, 3)
        colors = np.array(color_image).reshape(-1, 3) / 255.0

        # Create the point cloud and save it to the output directory
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud(
            os.path.join(
                args.outdir,
                os.path.splitext(os.path.basename(filename))[0] + ".ply",
            ),
            pcd,
        )


if __name__ == "__main__":
    main()
