"""Main script for video depth estimation."""

import argparse
import logging
import os

import cv2

from data_processing import get_video_path, setup_video_writer, depth_to_heatmap
from model_processing import load_model, process_frame

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(input_path: str, output_path: str, model_size: str):
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
        
    # Load model
    processor, model, device = load_model(model_size=model_size)

    # Get video path
    video_path = get_video_path(input_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info("Total frames: %s", total_frames)

    # Setup video writer
    base_output_path = output_path.rsplit('.', 1)[0]
    out_depth = setup_video_writer(cap, f"{base_output_path}_depth.mp4", is_color=False,)
    out_heatmap = setup_video_writer(cap, f"{base_output_path}_heatmap.mp4", is_color=True,)
    out_overlay = setup_video_writer(cap, f"{base_output_path}_overlay.mp4", is_color=True,)

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        depth_map = process_frame(frame_rgb, processor, model, device)

        # Convert depth to heatmap
        depth_heatmap = depth_to_heatmap(depth_map)
        
        # Normalize depth map to 0-255 range for video writing
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U,)
        
        # Create overlay of original frame and heatmap
        overlay = cv2.addWeighted(frame, 0.6, cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR), 0.4, 0,)

        # Convert heatmap back to BGR for video writing
        depth_heatmap_bgr = cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR)

        # Write the frames
        out_depth.write(depth_map_normalized)
        out_heatmap.write(cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR))
        out_overlay.write(overlay)

        frame_count += 1
        if frame_count % 100 == 0:
            logger.info("Processed %s frames | %s", frame_count, total_frames)

    # Release everything
    cap.release()
    out_depth.release()
    out_heatmap.release()
    out_overlay.release()

    logger.info("Depth map video saved to %s", f"{base_output_path}_depth.mp4")
    logger.info("Heatmap video saved to %s", f"{base_output_path}_heatmap.mp4")
    logger.info("Overlay video saved to %s", f"{base_output_path}_overlay.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Depth Estimation")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to input video file or folder containing videos",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        default="depth_video_output.mp4",
        help="Path to output video file",
    )
    parser.add_argument(
        "-m",
        "--model_size",
        choices=["small", "base", "large"],
        default="small",
        help="Size of the depth estimation model",
    )
    args = parser.parse_args()

    main(args.input, args.output, args.model_size)
