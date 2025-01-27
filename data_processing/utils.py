"""Module for video/image data processing."""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np


# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def depth_to_heatmap(
    depth: np.ndarray,
) -> np.ndarray:
    """
    Convert a grayscale depth image to a colored heatmap.

    Args:
        depth (np.ndarray): A 2D numpy array representing the grayscale depth image.

    Returns:
        np.ndarray: A 3D numpy array representing the colored heatmap image (RGB).
    """
    # Normalize the depth values to range [0, 1]
    depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

    # Apply the colormap
    cmap = plt.get_cmap("Spectral_r")
    heatmap = cmap(depth_normalized)

    # Convert to RGB and uint8
    heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)

    return heatmap_rgb


def load_frames(
    directory: str,
) -> tuple:
    """
    Load depth and RGB frames from the specified directory.

    Args:
        directory (os.PathLike): The directory containing the frames.

    Returns:
        Tuple[list, list]: A tuple containing the list of depth frames and RGB
        frames.
    """
    depth_frames = []
    rgb_frames = []
    for file in sorted(os.listdir(directory)):
        if file.endswith(".png"):
            full_path = os.path.join(directory, file)
            if "Depth" in file:
                depth_frames.append(full_path)
            elif "Frame" in file:
                rgb_frames.append(full_path)
    return (
        depth_frames,
        rgb_frames,
    )


def remove_bad_frames(
    rgb_list: list,
    depth_list: list,
    root_path: str,
) -> tuple:
    """
    Remove specific bad frames from the RGB and depth lists.

    Args:
        rgb_list (list): List of RGB frame paths
        depth_list (list): List of depth frame paths
        root_path (str): Root path of the dataset

    Returns:
        tuple: Updated RGB and depth lists
    """

    bad_frames = [
        "SyntheticColon_I/Frames_S14/FrameBuffer_0059.png",
        "SyntheticColon_I/Frames_S14/FrameBuffer_0060.png",
        "SyntheticColon_I/Frames_S14/FrameBuffer_0061.png",
    ]

    for frame in bad_frames:
        rgb_path = os.path.join(
            root_path,
            frame,
        )
        depth_path = os.path.join(
            root_path,
            frame.replace("FrameBuffer", "Depth"),
        )

        if rgb_path in rgb_list:
            rgb_list.remove(rgb_path)
        if depth_path in depth_list:
            depth_list.remove(depth_path)

    return (
        rgb_list,
        depth_list,
    )


def process_images(
    train_vids: str,
    val_vids: str,
    test_vids: str,
    input_path: str,
) -> tuple:
    """
    Process the input images and return the training and validation data.

    Args:
        input_path (str): The path to the input images.
    """

    train_depth = []
    train_rgb = []
    val_depth = []
    val_rgb = []
    test_depth = []
    test_rgb = []

    # Load training frames
    for vid in train_vids:
        depth, rgb = load_frames(vid)
        train_depth.extend(depth)
        train_rgb.extend(rgb)

    # Load validation frames
    for vid in val_vids:
        depth, rgb = load_frames(vid)
        val_depth.extend(depth)
        val_rgb.extend(rgb)

    # Load test frames
    for vid in test_vids:
        depth, rgb = load_frames(vid)
        test_depth.extend(depth)
        test_rgb.extend(rgb)

    # Remove bad frames from validation set
    val_rgb, val_depth = remove_bad_frames(
        val_rgb,
        val_depth,
        input_path,
    )

    assert len(train_depth) == len(train_rgb), "Mismatch in training data"
    assert len(val_depth) == len(val_rgb), "Mismatch in validation data"
    assert len(test_depth) == len(test_rgb), "Mismatch in validation data"

    logger.info(
        "Train size: %d, Val size: %d, Test size: %d",
        len(train_depth),
        len(val_depth),
        len(test_depth),
    )

    return (
        train_depth,
        train_rgb,
        val_depth,
        val_rgb,
        test_depth,
        test_rgb,
    )
