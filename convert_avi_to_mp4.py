"Module to convert AVI files to MP4"

import os
import argparse
import cv2


def convert_avi_to_mp4(input_dir: str, output_dir: str) -> None:
    """
    Convert AVI files in the input directory to MP4 format and save them in the output directory.

    Args:
        input_dir (str): Path to the directory containing AVI files.
        output_dir (str): Path to the directory where MP4 files will be saved.

    Raises:
        cv2.error: If OpenCV encounters an error during conversion.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".avi"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}.mp4"
            )

            # Open the AVI file
            cap = cv2.VideoCapture(input_path)

            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                else:
                    break

            # Release everything
            cap.release()
            out.release()

            print(f"Converted {filename} to MP4")


def main() -> None:
    """Parse command-line arguments and run the conversion process."""
    parser = argparse.ArgumentParser(
        description="Convert AVI files to MP4 format.",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        help="Path to the directory containing AVI files",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Path to the directory for output MP4 files",
    )
    args = parser.parse_args()

    convert_avi_to_mp4(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
