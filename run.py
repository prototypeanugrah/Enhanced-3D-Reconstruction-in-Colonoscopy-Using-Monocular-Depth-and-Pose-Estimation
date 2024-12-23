"""
Model to generate depth maps using the DepthAnythingV2 model
"""

import argparse
import os
import glob

from pathlib import Path
from tqdm import tqdm
import cv2
import matplotlib
import numpy as np
import torch

from Depth_Anything_V2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Depth Anything V2 Metric Depth Estimation"
    )

    parser.add_argument("--img-path", type=str)
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--outdir", type=str)

    parser.add_argument(
        "--encoder",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl", "vitg"],
    )
    parser.add_argument(
        "--load-from",
        type=str,
        default="checkpoints/depth_anything_v2_metric_hypersim_vitl.pth",
    )
    parser.add_argument("--max-depth", type=float, default=20)

    parser.add_argument(
        "--save-numpy",
        dest="save_numpy",
        action="store_true",
        help="save the model raw output",
    )
    parser.add_argument(
        "--pred-only",
        dest="pred_only",
        action="store_true",
        help="only display the prediction",
    )
    parser.add_argument(
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="do not apply colorful palette",
    )

    args = parser.parse_args()

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

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
    depth_anything = depth_anything.to(DEVICE).eval()

    filenames = []
    if os.path.isfile(args.img_path):
        if args.img_path.endswith("txt"):
            with open(args.img_path, "r") as f:
                filenames = f.read().splitlines()
        else:
            # Single image processing
            filenames = [args.img_path]
            if args.outdir is None:
                args.outdir = str(Path(args.img_path).parent)
    else:
        # SimCol dataset processing
        base_dir = Path(args.img_path)
        for suffix in ["I", "II", "III"]:
            pattern = f"SyntheticColon_{suffix}/Frames_*/FrameBuffer_*.png"
            filenames.extend(
                glob.glob(
                    str(base_dir / pattern),
                    recursive=True,
                )
            )
        if args.outdir is None:
            args.outdir = str(base_dir)

    # Create base output directory
    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap("Spectral")

    skipped = 0
    # for k, filename in enumerate(filenames):
    for filename in tqdm(
        filenames,
        desc="Processing images",
        unit="image",
    ):

        raw_image = cv2.imread(filename)

        depth = depth_anything.infer_image(raw_image, args.input_size)

        if os.path.isfile(args.img_path):
            # Single image - save directly in outdir
            base_name = Path(filename).stem
            output_folder = Path(args.outdir)
        else:
            # SimCol dataset - maintain directory structure but with _OP suffix
            rel_path = Path(filename).relative_to(Path(args.img_path))
            parent_folder = rel_path.parent
            frames_dir = parent_folder.name  # e.g., "Frames_O1"
            output_folder = (
                Path(args.img_path) / parent_folder.parent / f"{frames_dir}_P"
            )
            base_name = Path(filename).stem

        # Check if files already exist
        npy_path = output_folder / f"{base_name}.npy"
        png_path = output_folder / f"{base_name}.png"

        if npy_path.exists() and png_path.exists():
            skipped += 1
            continue

        # Process image only if files don't exist
        raw_image = cv2.imread(filename)
        depth = depth_anything.infer_image(raw_image, args.input_size)

        output_folder.mkdir(parents=True, exist_ok=True)

        # Save raw depth in meters
        if args.save_numpy:
            # output_path = output_folder / f"{base_name}.npy"
            np.save(str(npy_path), depth)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)

        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if args.pred_only:
            cv2.imwrite(str(png_path), depth)
        else:
            split_region = (
                np.ones(
                    (raw_image.shape[0], 50, 3),
                    dtype=np.uint8,
                )
                * 255
            )
            combined_result = cv2.hconcat([raw_image, split_region, depth])

            cv2.imwrite(str(png_path), combined_result)

    print(f"\nProcessing complete:")
    print(f"- Total files: {len(filenames)}")
    print(f"- Skipped existing: {skipped}")
    print(f"- Newly processed: {len(filenames) - skipped}")


# python run.py --encoder vitl --load-from "/home/public/avaishna/Endoscopy-3D-Modeling/checkpoints/simcol/mvitl_l5e-06_b20_e30_dsimcol/depth-any-endoscopy-epoch=28-val_loss=0.01.ckpt" --max-depth 20 --img_path datasets/SyntheticColon/
