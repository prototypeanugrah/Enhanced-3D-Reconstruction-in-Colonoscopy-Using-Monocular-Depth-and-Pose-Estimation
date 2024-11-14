"""Module for depth estimation model processing."""

import torch

from lightning_model import DepthEstimationModule
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from typing import Literal


def load_model(model_size: Literal["small", "base", "large"] = "small"):
    """
    Load the DepthAnything model and processor.

    Args:
        model_size (Literal["small", "base", "large"]): Size of the model to
        load. Defaults to "small".

    Returns:
        Tuple[AutoImageProcessor, AutoModelForDepthEstimation, str]: A tuple
        containing:
            - The image processor
            - The depth estimation model
            - The device (either "cuda" or "cpu")
    """
    # Check if CUDA is available, otherwise use CPU. Construct the model name based on the selected size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"depth-anything/Depth-Anything-V2-{model_size}-hf"

    # Load the image processor and model
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device)

    return processor, model, device


def load_model_lightning(model_size: Literal["small", "base", "large"] = "small"):
    """
    Load the DepthAnything model and processor.

    Args:
        model_size (Literal["small", "base", "large"]): Size of the model to load. Defaults to "small".

    Returns:
        Tuple[AutoImageProcessor, AutoModelForDepthEstimation, str]: A tuple containing:
            - The image processor
            - The depth estimation model
            - The device (either "cuda" or "cpu")
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"depth-anything/Depth-Anything-V2-{model_size}-hf"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = DepthEstimationModule(model_name)

    return processor, model, device


def process_frame(frame, processor, model, device):
    """
    Process a single frame through the depth estimation model.

    Args:
        frame (torch.Tensor): The input frame to process.
        processor (AutoImageProcessor): The image processor for the model.
        model (AutoModelForDepthEstimation): The depth estimation model.
        device (str): The device to run the model on ("cuda" or "cpu").

    Returns:
        np.ndarray: The processed depth map as a numpy array.
    """
    # Shape: (1, C, H, W) where 1 is the batch size
    inputs = processor(images=frame, return_tensors="pt").to(device)

    # Run the model without gradient computation
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate the predicted depth to match the input frame size
    # Shape: (1, 1, H, W)
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    )

    # Remove batch and channel dimensions, convert to numpy array
    # Final shape: (H, W)
    return prediction.squeeze().cpu().numpy()


def process_frame_lightning(frame, processor, model, device):
    """
    Process a single frame through the depth estimation model.

    Args:
        frame (torch.Tensor): The input frame to process.
        processor (AutoImageProcessor): The image processor for the model.
        model (AutoModelForDepthEstimation): The depth estimation model.
        device (str): The device to run the model on ("cuda" or "cpu").

    Returns:
        np.ndarray: The processed depth map as a numpy array.
    """

    # Process the frame
    inputs = processor(images=frame, return_tensors="pt").to(device)

    # Run inference
    with torch.no_grad():
        outputs = model(inputs["pixel_values"])

    # Extract the predicted depth
    predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=frame.shape[:2],
        mode="bicubic",
        align_corners=False,
    )

    # Convert to numpy array
    output = prediction.squeeze().cpu().numpy()

    return output
