"""Main script for video depth estimation."""

import argparse
import logging
import os

import cv2

from data_processing import get_video_paths, setup_video_writer, depth_to_heatmap
from model_processing import load_model, process_frame
from tqdm import tqdm

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
os.environ['HF_HOME'] = '~/home/public/avaishna/.cache'


def process_video(video_path: str, output_path: str, processor, model, device):
    # Get input filename without extension
    input_filename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output directory
    output_dir = os.path.join(output_path, input_filename)
    os.makedirs(output_dir, exist_ok=True)
        
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup video writers
    out_depth = setup_video_writer(cap, os.path.join(output_dir, f"{input_filename}_depth.mp4"), is_color=False)
    out_heatmap = setup_video_writer(cap, os.path.join(output_dir, f"{input_filename}_heatmap.mp4"), is_color=True)
    out_overlay = setup_video_writer(cap, os.path.join(output_dir, f"{input_filename}_overlay.mp4"), is_color=True)

    # Process frames with tqdm progress bar
    # for _ in tqdm(range(total_frames), desc=f"Processing {input_filename}", unit="frame"):
    for _ in range(total_frames):
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
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Create overlay of original frame and heatmap
        overlay = cv2.addWeighted(frame, 0.6, cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR), 0.4, 0)

        # Write the frames
        out_depth.write(depth_map_normalized)
        out_heatmap.write(cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR))
        out_overlay.write(overlay)

    # Release everything
    cap.release()
    out_depth.release()
    out_heatmap.release()
    out_overlay.release()

    # logger.info(f"Processed video: {video_path}")
    
    
def main(input_path: str, output_path: str, model_size: str, limit: int = None):
    # Load model
    processor, model, device = load_model(model_size=model_size)

    # Get video paths
    video_paths = get_video_paths(input_path)
    
    # Limit the number of videos if specified
    if limit is not None:
        video_paths = video_paths[:limit]
        logger.info(f"Processing the first {limit} videos")
    
    # Process videos with tqdm progress bar
    for video_path in tqdm(video_paths, desc="Processing videos", unit="video"):
        process_video(video_path, output_path, processor, model, device)
# def main(input_path: str, output_path: str, model_size: str):
    
#     # Get video path
#     video_path = get_video_path(input_path)
    
#     # Get input filename without extension
#     input_filename = os.path.splitext(os.path.basename(video_path))[0]
    
#     # Create output directory
#     output_dir = os.path.join(output_path, input_filename)
#     os.makedirs(output_dir, exist_ok=True)
        
#     # Load model
#     processor, model, device = load_model(model_size=model_size)

#     # Open the video file
#     cap = cv2.VideoCapture(video_path)

#     # Get total frame count
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Setup video writer
#     out_depth = setup_video_writer(cap, os.path.join(output_dir, f"{input_filename}_depth.mp4"), is_color=False)
#     out_heatmap = setup_video_writer(cap, os.path.join(output_dir, f"{input_filename}_heatmap.mp4"), is_color=True)
#     out_overlay = setup_video_writer(cap, os.path.join(output_dir, f"{input_filename}_overlay.mp4"), is_color=True)

#     frame_count = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Convert frame to RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Process frame
#         depth_map = process_frame(frame_rgb, processor, model, device)

#         # Convert depth to heatmap
#         depth_heatmap = depth_to_heatmap(depth_map)
        
#         # Normalize depth map to 0-255 range for video writing
#         depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U,)
        
#         # Create overlay of original frame and heatmap
#         overlay = cv2.addWeighted(frame, 0.6, cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR), 0.4, 0,)

#         # Convert heatmap back to BGR for video writing
#         depth_heatmap_bgr = cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR)

#         # Write the frames
#         out_depth.write(depth_map_normalized)
#         out_heatmap.write(cv2.cvtColor(depth_heatmap, cv2.COLOR_RGB2BGR))
#         out_overlay.write(overlay)

#         frame_count += 1
#         if frame_count % 100 == 0:
#             logger.info("Processed %s frames | Total frames: %s", frame_count, total_frames)

#     # Release everything
#     cap.release()
#     out_depth.release()
#     out_heatmap.release()
#     out_overlay.release()
    
#     # Check if output files exist
#     for suffix in ["depth", "heatmap", "overlay"]:
#         output_file = os.path.join(output_dir, f"{input_filename}_{suffix}.mp4")
#         if os.path.exists(output_file):
#             logger.info(f"{suffix.capitalize()} video saved to: {output_file}")
#         else:
#             logger.error(f"{suffix.capitalize()} video not found at: {output_file}")


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
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit the number of videos to process",
    )
    args = parser.parse_args()

    main(args.input, args.output, args.model_size, args.limit)
