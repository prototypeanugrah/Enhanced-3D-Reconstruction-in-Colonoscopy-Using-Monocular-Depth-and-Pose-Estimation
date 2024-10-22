"""Module for video/image data processing."""

import logging
import os

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils import data

import cv2
import matplotlib.pyplot as plt
import numpy as np

import custom_dataset_frames

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_video_paths(input_path: str) -> str:
    """
    Get the path of the video file or the first .avi video in the specified
    folder.

    Args:
        input_path (str): The path to the video file or folder containing video
        files.

    Returns:
        str: The full path to the video file.

    Raises:
        ValueError: If no video files are found or the input path is invalid.
    """
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        video_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith((".avi", ".mp4"))
        ]
        if not video_files:
            raise ValueError("No video files found in the specified folder")
        return video_files
    else:
        raise ValueError(
            "Invalid input path. Please provide a valid video file or folder."
        )


def setup_video_writer(
    cap: cv2.VideoCapture, output_path: str, is_color: bool = True
) -> cv2.VideoWriter:
    """
    Set up VideoWriter for the output video.

    Args:
        cap (cv2.VideoCapture): The video capture object of the input video.
        output_path (str): The path where the output video will be saved.
        is_color (bool): Whether the output video should be in color. Default is True.

    Returns:
        cv2.VideoWriter: A VideoWriter object configured for the output video.
    """
    # Get video properties from the input video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    return cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=is_color)


def depth_to_heatmap(depth: np.ndarray) -> np.ndarray:
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
    cmap = plt.get_cmap("plasma")
    heatmap = cmap(depth_normalized)

    # Convert to RGB and uint8
    heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)

    return heatmap_rgb


def load_frames(directory):
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
    return depth_frames, rgb_frames


def process_images(input_path: str):
    """
    Process the input images and return the training and validation data.

    Args:
        input_path (str): The path to the input images.
    """
    train_vids = [
        os.path.join(input_path, "Frames_O1"),
        os.path.join(input_path, "Frames_O2"),
    ]
    val_vids = [
        os.path.join(input_path, "Frames_O3"),
    ]

    train_depth = []
    train_rgb = []
    val_depth = []
    val_rgb = []

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

    assert len(train_depth) == len(train_rgb), "Mismatch in training data"
    assert len(val_depth) == len(val_rgb), "Mismatch in validation data"

    logger.info("Train size: %d, Val size: %d", len(train_depth), len(val_depth))

    return train_depth, train_rgb, val_depth, val_rgb


def split_ids(len_ids):
    """
    Split the IDs into training, validation, and testing sets.

    Args:
        len_ids (float): The length of the IDs.

    Returns:
        tuple: A tuple containing the training, validation, and testing indices.
    """
    train_size = int(round((80 / 100) * len_ids))
    valid_size = int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))

    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )

    train_indices, val_indices = train_test_split(
        train_indices, test_size=test_size, random_state=42
    )

    return train_indices, test_indices, val_indices


def get_dataloaders(train_maps, train_imgs, val_maps, val_imgs, batch_size):
    """
    Get the training and validation dataloaders.

    Args:
        train_maps (depth maps): The training depth maps.
        train_imgs (images): The training images.
        val_maps (depth maps): The validation depth maps.
        val_imgs (images): The validation images.
        batch_size (int): The batch size.

    Returns:
        tuple: A tuple containing the training and validation dataloaders.
    """
    transform_input = transforms.Compose(
        [
            transforms.Resize((352, 352)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    transform_target = transforms.ToTensor()

    train_dataset = custom_dataset_frames.Dataset(
        input_paths=train_imgs,
        target_paths=train_maps,
        transform_input=transform_input,
        transform_target=transform_target,
        hflip=True,
        vflip=True,
        affine=False,
    )

    val_dataset = custom_dataset_frames.Dataset(
        input_paths=val_imgs,
        target_paths=val_maps,
        transform_input=transform_input,
        transform_target=transform_target,
    )

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )

    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
    )

    return train_dataloader, val_dataloader
