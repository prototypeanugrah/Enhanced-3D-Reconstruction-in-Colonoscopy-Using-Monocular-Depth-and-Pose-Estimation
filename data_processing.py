"""Module for video data processing."""

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_video_path(input_path: str) -> str:
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
        return input_path
    elif os.path.isdir(input_path):
        video_files = [
            f for f in os.listdir(input_path) if f.endswith((".avi", ".mp4"))
        ]
        if not video_files:
            raise ValueError("No video files found in the specified folder")
        return os.path.join(input_path, video_files[0])
    else:
        raise ValueError(
            "Invalid input path. Please provide a valid video file or folder."
        )


def setup_video_writer(cap: cv2.VideoCapture, output_path: str) -> cv2.VideoWriter:
    """
    Set up VideoWriter for the output video.

    Args:
        cap (cv2.VideoCapture): The video capture object of the input video.
        output_path (str): The path where the output video will be saved.

    Returns:
        cv2.VideoWriter: A VideoWriter object configured for the output video.
    """
    # Get video properties from the input video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


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
