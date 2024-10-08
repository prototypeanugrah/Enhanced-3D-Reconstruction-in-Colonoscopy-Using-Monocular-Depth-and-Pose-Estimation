"""Main script for video depth estimation."""

import argparse

import cv2

from data_processing import get_video_path, setup_video_writer, depth_to_heatmap
from model_processing import load_model, process_frame


def main(input_path: str, output_path: str, model_size: str):
    # Load model
    processor, model, device = load_model(model_size=model_size)

    # Get video path
    video_path = get_video_path(input_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writer
    out = setup_video_writer(cap, output_path)

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

        # Convert heatmap back to BGR for video writing
        depth_heatmap_bgr = cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(depth_heatmap_bgr)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames | {total_frames}")

    # Release everything
    cap.release()
    out.release()

    print(f"Depth estimation video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Depth Estimation")
    parser.add_argument(
        "input", help="Path to input video file or folder containing videos"
    )
    parser.add_argument(
        "--output", default="depth_video_output.mp4", help="Path to output video file"
    )
    parser.add_argument(
        "--model_size",
        choices=["small", "base", "large"],
        default="small",
        help="Size of the depth estimation model",
    )
    args = parser.parse_args()

    main(args.input, args.output, args.model_size)
