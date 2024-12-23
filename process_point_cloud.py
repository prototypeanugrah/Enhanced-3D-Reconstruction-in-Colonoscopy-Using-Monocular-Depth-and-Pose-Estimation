import argparse
import glob
import os

from pathlib import Path
from tqdm import tqdm

from point_cloud import process_sequence


def main(
    base_dir: Path,
):
    """
    Process all frames across SyntheticColon_I, II, and III

    Args:
        base_dir: Base directory containing all SyntheticColon datasets
    """
    # Process each SyntheticColon dataset (I, II, III)
    for suffix in ["I", "II", "III"]:
        print(f"\nProcessing SyntheticColon_{suffix}")

        # Find all RGB images
        pattern = f"SyntheticColon_{suffix}/Frames_*/FrameBuffer_*.png"
        rgb_files = glob.glob(str(base_dir / pattern), recursive=True)

        pbar = tqdm(
            sorted(rgb_files),
            desc=f"SyntheticColon_{suffix}",
            unit="frame",
        )

        for rgb_path in pbar:
            rgb_path = Path(rgb_path)

            # Extract components from path
            colon_dir = rgb_path.parents[1]  # SyntheticColon_X directory
            frames_dir = rgb_path.parent.name  # Frames_XX
            frame_num = rgb_path.name.split("_")[1].split(".")[
                0
            ]  # XXXX from FrameBuffer_XXXX.png

            # Construct related paths
            depth_path = colon_dir / f"{frames_dir}_OP" / "depth" / rgb_path.name
            output_dir = colon_dir / f"{frames_dir}_PC"
            output_path = output_dir / f"pc_{frame_num}"

            # Check if point cloud file already exists
            if (output_path.with_suffix(".ply")).exists():
                pbar.set_postfix(
                    {
                        "Status": "Exists",
                        "Dir": frames_dir,
                        "Frame": rgb_path.name,
                    }
                )
                continue

            pbar.set_postfix(
                {
                    "Status": "Processing",
                    "Dir": frames_dir,
                    "Frame": rgb_path.name,
                }
            )

            # Get camera and pose files
            cam_file = colon_dir / "cam.txt"
            sequence_num = frames_dir.split("_")[1]  # XX from Frames_XX
            position_file = colon_dir / f"SavedPosition_{sequence_num}.txt"
            quaternion_file = colon_dir / f"SavedRotationQuaternion_{sequence_num}.txt"

            # Create output directory
            output_dir.mkdir(exist_ok=True)

            try:
                process_sequence(
                    rgb_path=str(rgb_path),
                    depth_path=str(depth_path),
                    cam_file=str(cam_file),
                    pose_file=None,  # Not needed for SimCol3D
                    output_path=str(output_path),
                    dataset_type="simcol3d",
                    position_file=str(position_file),
                    quaternion_file=str(quaternion_file),
                )
            except Exception as e:
                tqdm.write(f"Error processing frame {frame_num}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process SimCol3D datasets",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        help="Base directory containing SyntheticColon_I, II, III directories",
        type=Path,
        required=True,
    )
    args = parser.parse_args()
    main(args.input_dir)

# python process_point_cloud.py -i datasets/SyntheticColon/
