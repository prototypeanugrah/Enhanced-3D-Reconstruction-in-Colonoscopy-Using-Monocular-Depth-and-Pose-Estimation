"""
Script to convert mp4 videos to frame images
"""

from datetime import datetime
import argparse
import os
import yaml

from tqdm import tqdm
import cv2


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_used_config(config: dict, output_path: str):
    """Save the configuration used for this run"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = os.path.join(output_path, f"config_used_{timestamp}.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Convert video to frames with cropping"
    )
    parser.add_argument(
        "--config",
        default="./configs/video_to_image_config.yaml",
        help="Path to config file",
    )
    parser.add_argument("--input_path", help="Input video path")
    parser.add_argument("--output_path", help="Output directory path")
    parser.add_argument("--start_time", type=float, help="Start time in seconds")
    parser.add_argument("--end_time", type=float, help="End time in seconds")
    parser.add_argument("--x", type=int, help="Crop x coordinate")
    parser.add_argument("--y", type=int, help="Crop y coordinate")
    parser.add_argument("--width", type=int, help="Crop width")
    parser.add_argument("--height", type=int, help="Crop height")
    parser.add_argument("--test", action="store_true", help="Test mode")
    parser.add_argument(
        "--test_frames",
        type=int,
        default=10,
        help="# frames to extract in test mode",
    )
    return parser.parse_args()


def video_to_frames(
    video_path: str,
    output_path: str,
    start_time: float,  # in seconds
    end_time: float,  # in seconds
    crop_coords: tuple,  # (x, y, width, height)
    test_mode: bool,
    test_frames: int,
):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Read the video
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Set default end time if not specified
    if end_time is None:
        end_time = duration

    # Calculate start and end frames
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Set video position to start_frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Initialize frame counter
    frame_count = 0

    total_frames = test_frames if test_mode else (end_frame - start_frame)

    # Create progress bar
    with tqdm(
        total=total_frames,
        desc="Extracting frames",
    ) as pbar:
        while True:
            success, frame = video.read()

            if not success or frame_count >= (end_frame - start_frame):
                break

            # In test mode, only process specified number of frames
            if test_mode and frame_count >= test_frames:
                break

            # Crop the frame if crop coordinates are provided
            if crop_coords:
                x, y, w, h = crop_coords
                frame = frame[y : y + h, x : x + w]

            # Save frame as image
            frame_path = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)

            frame_count += 1
            pbar.update(1)

    # Release video object
    video.release()
    print(f"\nExtracted {frame_count} frames to {output_path}")

    # Print additional info in test mode
    if test_mode:
        print("\nTest mode summary:")
        print(f"Start time: {start_time} seconds")
        print(
            f"Crop coordinates: x={crop_coords[0]}, y={crop_coords[1]}, "
            f"width={crop_coords[2]}, height={crop_coords[3]}"
        )
        print(f"Output directory: {output_path}")
        print("\nTo run the full extraction, run the script without --test flag")


if __name__ == "__main__":

    args = parse_arguments()
    config = load_config(args.config)

    # Override config with command line arguments if provided
    for key, value in vars(args).items():
        if value is not None and key not in ["config", "test", "test_frames"]:
            config[key] = value

    # Create output directory if it doesn't exist
    os.makedirs(config["output_path"], exist_ok=True)

    if args.test:
        print("\nRunning in test mode...")
        output_dir = os.path.join(config["output_path"], "test")
        os.makedirs(output_dir, exist_ok=True)
        # save_used_config(config, output_dir)
    else:
        output_dir = config["output_path"]
        # save_used_config(config, config["output_path"])

    # Extract frames
    video_to_frames(
        video_path=config["input_path"],
        output_path=output_dir,
        start_time=config["start_time"],
        end_time=config["end_time"],
        crop_coords=(
            config["x"],
            config["y"],
            config["width"],
            config["height"],
        ),
        test_mode=args.test,
        test_frames=args.test_frames,
    )
