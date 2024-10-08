"""
Module for depth estimation using DepthAnything model from Transformers.
"""

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def depth_to_heatmap(depth: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale depth image to a colored heatmap.

    Args:
        depth (np.ndarray): Grayscale depth image.

    Returns:
        np.ndarray: RGB heatmap image.
    """
    depth_normalized = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    cmap = plt.get_cmap("plasma")
    heatmap = cmap(depth_normalized)
    heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)
    return heatmap_rgb


DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

model_size: Literal["small", "base", "large"] = "small"
model_name: str = f"depth-anything/Depth-Anything-V2-{model_size}-hf"

# Load model and processor
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForDepthEstimation.from_pretrained(model_name).to(DEVICE)

# ### Images ###
# # Load and process image
# image_path: str = "/Users/anugrahvaishnav/Downloads/IMG_4582.jpg"
# raw_image: np.ndarray = cv2.imread(image_path)
# raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

# # Prepare image for the model
# inputs = processor(images=raw_image, return_tensors="pt").to(DEVICE)

# # Perform inference
# with torch.no_grad():
#     outputs = model(**inputs)
#     predicted_depth = outputs.predicted_depth

# # Interpolate to original size
# prediction = torch.nn.functional.interpolate(
#     predicted_depth.unsqueeze(1),
#     size=raw_image.shape[:2],
#     mode="bicubic",
#     align_corners=False,
# ).squeeze()

# depth_map = prediction.cpu().numpy()

# # Convert depth to heatmap
# depth_heatmap: np.ndarray = depth_to_heatmap(depth_map)

# # Display the heatmap
# plt.figure(figsize=(10, 10))
# plt.imshow(depth_heatmap)
# plt.axis("off")
# plt.title("Depth Heatmap")
# plt.show()


### Videos ###
# Get the first video in the folder
video_folder = "hyper-kvasir-videos/videos"
video_files = [f for f in os.listdir(video_folder) if f.endswith(".avi")]
if not video_files:
    raise ValueError("No .avi files found in the specified folder")
video_path = os.path.join(video_folder, video_files[0])

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object to save the output video
output_path = "depth_video_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare image for the model
    inputs = processor(images=frame_rgb, return_tensors="pt").to(DEVICE)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=frame_rgb.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Convert depth to heatmap
    depth_heatmap: np.ndarray = depth_to_heatmap(depth_map)

    # Convert heatmap back to BGR for video writing
    depth_heatmap_bgr = cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR)

    # Write the frame
    out.write(depth_heatmap_bgr)

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames")

# Release everything
cap.release()
out.release()

print(f"Depth estimation video saved to {output_path}")
