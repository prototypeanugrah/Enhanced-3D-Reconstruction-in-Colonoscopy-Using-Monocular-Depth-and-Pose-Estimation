"""
Script to convert mp4 videos to frame images
"""

import cv2
import os
from tqdm import tqdm


def video_to_frames(
    video_path: str,
    output_path: str,
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

    # Get total frame count
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize frame counter
    frame_count = 0

    # Create progress bar
    with tqdm(
        total=total_frames,
        desc="Extracting frames",
    ) as pbar:
        while True:
            # Read a frame
            success, frame = video.read()

            # Break the loop if we can't read any more frames
            if not success:
                break

            # Save frame as image
            frame_path = os.path.join(output_path, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)

            frame_count += 1
            pbar.update(1)

    # Release video object
    video.release()
    print(f"\nExtracted {frame_count} frames to {output_path}")


if __name__ == "__main__":
    # Example usage
    video_path = "/data1_ycao/public/share_data_Anugrah/colonoscopy_videos/20220107_160140_01_c.mp4"  # Replace with your video path
    output_path = "./convert/test_video/colonoscopy_videos/20220107_160140_01_c/"  # Replace with your output directory path

    video_to_frames(video_path, output_path)
