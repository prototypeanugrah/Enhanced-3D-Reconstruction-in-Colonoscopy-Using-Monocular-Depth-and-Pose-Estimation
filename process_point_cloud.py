import argparse
import glob
import traceback

from pathlib import Path
from tqdm import tqdm
import open3d as o3d

from point_cloud import process_sequence


def main(
    base_dir: Path,
    batch_size: int = 50,
    voxel_size: float = 0.0001,
    pred: bool = True,
    debug: bool = False,
):
    """Process all frames across SyntheticColon_I, II, and III"""
    for suffix in [
        "I",
        # "II",
        # "III",
    ]:
        print(f"\nProcessing SyntheticColon_{suffix}")

        # Group files by sequence
        sequence_files = {}
        pattern = f"SyntheticColon_{suffix}/Frames_S1/FrameBuffer_*.png"
        rgb_files = sorted(glob.glob(str(base_dir / pattern), recursive=True))

        for rgb_path in rgb_files:
            rgb_path = Path(rgb_path)
            sequence = rgb_path.parent.name  # e.g., "Frames_S1"
            if sequence not in sequence_files:
                sequence_files[sequence] = []
            sequence_files[sequence].append(rgb_path)

        # Process each sequence separately
        for sequence, files in sequence_files.items():
            print(f"\nProcessing sequence {sequence}")

            # Initialize storage for intermediate results
            temp_dir = base_dir / f"SyntheticColon_{suffix}" / sequence / "temp_batches"
            temp_dir.mkdir(exist_ok=True, parents=True)

            # Process in batches
            current_batch = []
            batch_count = 0

            pbar = tqdm(
                files,
                desc=f"{sequence}",
                unit="frame",
            )

            for rgb_path in pbar:
                rgb_path = Path(rgb_path)

                # Extract components from path
                colon_dir = rgb_path.parents[1]  # SyntheticColon_X directory
                frames_dir = rgb_path.parent.name  # Frames_XX
                frame_num = rgb_path.name.split("_")[1].split(".")[0]

                # Construct related paths
                if pred:
                    depth_path = (
                        colon_dir / f"{frames_dir}_OP" / f"Depth_{frame_num}.png"
                    )  # Predicted depth map
                    output_dir = colon_dir / f"{frames_dir}_PC"  # Predicted ply
                else:
                    depth_path = (
                        colon_dir / f"{frames_dir}" / f"Depth_{frame_num}.png"
                    )  # OG depth map
                    output_dir = colon_dir / f"{frames_dir}_PC_OG"  # OG ply
                output_path = output_dir / f"pc_{frame_num}"

                # Get camera and pose files
                cam_file = colon_dir / "cam.txt"
                sequence_num = frames_dir.split("_")[1]
                position_file = colon_dir / f"SavedPosition_{sequence_num}.txt"
                quaternion_file = (
                    colon_dir / f"SavedRotationQuaternion_{sequence_num}.txt"
                )

                output_dir.mkdir(exist_ok=True)

                try:
                    frame_pcd = process_sequence(
                        rgb_path=str(rgb_path),
                        depth_path=str(depth_path),
                        cam_file=str(cam_file),
                        pose_file=None,
                        output_path=str(output_path),
                        dataset_type="simcol3d",
                        position_file=str(position_file),
                        quaternion_file=str(quaternion_file),
                    )

                    if debug:
                        exit(0)

                    if frame_pcd is None:
                        exit(0)

                    current_batch.append(frame_pcd)

                    if len(current_batch) >= batch_size:
                        batch_count += 1
                        pbar.set_postfix({"Processing": f"Batch {batch_count}"})

                        processed_batch = process_batch(
                            current_batch, voxel_size=voxel_size
                        )

                        batch_path = temp_dir / f"batch_{batch_count:03d}.ply"
                        o3d.io.write_point_cloud(str(batch_path), processed_batch)

                        current_batch = []

                except Exception as e:
                    tqdm.write(f"\nError processing frame {frame_num}:")
                    tqdm.write(traceback.format_exc())
                    continue

            # Process remaining frames in last batch
            if current_batch:
                batch_count += 1
                processed_batch = process_batch(current_batch, voxel_size=voxel_size)
                batch_path = temp_dir / f"batch_{batch_count:03d}.ply"
                o3d.io.write_point_cloud(str(batch_path), processed_batch)

            # Combine all batches for this sequence
            print(f"\nCombining all batches for {sequence}")
            final_combined = o3d.geometry.PointCloud()

            for batch_file in sorted(temp_dir.glob("batch_*.ply")):
                batch_pcd = o3d.io.read_point_cloud(str(batch_file))
                final_combined += batch_pcd

            # Final processing on combined result
            if voxel_size > 0.0:
                final_combined = final_combined.voxel_down_sample(voxel_size)

            # Force a minimum voxel size for the final save to prevent file size issues
            if len(final_combined.points) > 10000000:  # If more than 10 million points
                save_voxel_size = max(
                    voxel_size, 0.00005
                )  # Minimum 0.05mm voxel size for saving
                save_cloud = final_combined.voxel_down_sample(save_voxel_size)
            else:
                save_cloud = final_combined
            # final_combined, _ = final_combined.remove_statistical_outlier(
            #     nb_neighbors=30,
            #     std_ratio=3.0,
            # )

            # Save final result for this sequence
            if pred:
                output_path = (
                    base_dir
                    / f"SyntheticColon_{suffix}"
                    / f"{sequence}_PC"
                    / "combined_scene.ply"
                )
            else:
                output_path = (
                    base_dir
                    / f"SyntheticColon_{suffix}"
                    / f"{sequence}_PC_OG"
                    / "combined_scene_og.ply"
                )
            o3d.io.write_point_cloud(str(output_path), save_cloud)

            # Cleanup temporary files
            if temp_dir.exists():
                for batch_file in temp_dir.glob("batch_*.ply"):
                    batch_file.unlink()
                temp_dir.rmdir()

            print(
                f"Final point cloud for {sequence} saved with {len(final_combined.points)} points"
            )


def process_batch(
    pcd_batch: list,
    voxel_size: float = 0.001,
    clean_outliers: bool = False,
) -> o3d.geometry.PointCloud:
    """Process a batch of point clouds, combining and cleaning them"""

    # # Print initial points
    # total_points = sum(len(pcd.points) for pcd in pcd_batch)
    # print(f"\nBatch processing stats:")
    # print(f"- Initial points: {total_points}")

    # Combine batch
    combined = o3d.geometry.PointCloud()
    for pcd in pcd_batch:
        # Downsample to remove duplicates
        if voxel_size > 0.0:
            combined = combined.voxel_down_sample(voxel_size)
        combined += pcd

    # Optional outlier removal
    if clean_outliers:
        combined, _ = combined.remove_statistical_outlier(
            nb_neighbors=30,
            std_ratio=3.0,
        )
        # print(f"- Points after outlier removal: {len(combined.points)}")

    return combined


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
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Number of frames to process in each batch",
        type=int,
        default=50,
    )
    parser.add_argument(
        "-v",
        "--voxel_size",
        help="Voxel size for downsampling (in meters)",
        type=float,
        # default=0.00005,
        default=0.0,
    )
    parser.add_argument(
        "-p",
        "--pred_depth",
        help="To/Not to use pred depth map for point cloud generation",
        type=bool,
        default=False,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Whether in debug phase",
        type=bool,
        default=False,
    )
    args = parser.parse_args()
    main(
        args.input_dir,
        args.batch_size,
        args.voxel_size,
        args.pred_depth,
        args.debug,
    )

# python process_point_cloud.py -i datasets/SyntheticColon/ -b 100 -p True
