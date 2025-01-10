# from pathlib import Path
# from tqdm import tqdm
# import glob
# import numpy as np
# import open3d as o3d


# # Function to process a batch of point clouds
# def process_batch(batch_files):
#     # Initialize variables
#     combined_batch_pcd = None
#     processed_pcds = []

#     # First pass: load and preprocess all PCDs in batch
#     for file in tqdm(batch_files, desc="Processing batch"):
#         pcd = o3d.io.read_point_cloud(file)

#         # Denoise and estimate normals
#         pcd, _ = pcd.remove_statistical_outlier(
#             nb_neighbors=20,
#             std_ratio=2.0,
#         )
#         pcd.estimate_normals(
#             search_param=o3d.geometry.KDTreeSearchParamHybrid(
#                 radius=0.01,
#                 max_nn=30,
#             )
#         )
#         processed_pcds.append(pcd)

#     # Second pass: align all PCDs to the first one
#     base_pcd = processed_pcds[0]
#     aligned_pcds = [base_pcd]

#     for source in tqdm(processed_pcds[1:], desc="Aligning PCDs"):
#         # Perform ICP alignment
#         result = o3d.pipelines.registration.registration_icp(
#             source,
#             base_pcd,
#             max_correspondence_distance=0.05,
#             estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#             criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
#                 max_iteration=50
#             ),
#         )

#         # Apply the transformation
#         source.transform(result.transformation)
#         aligned_pcds.append(source)

#     # Combine all aligned PCDs
#     combined_batch_pcd = aligned_pcds[0]
#     for pcd in aligned_pcds[1:]:
#         combined_batch_pcd += pcd

#     # Downsample the combined batch point cloud
#     combined_batch_pcd = combined_batch_pcd.voxel_down_sample(voxel_size=0.005)
#     return combined_batch_pcd


# # Step 1: Load Point Clouds for Each Frame
# suffix = "I"
# pattern = f"SyntheticColon_{suffix}/Frames_S1_PC_OG/pc_*.ply"
# base_dir = Path("./datasets/SyntheticColon/")
# pc_og_files = sorted(
#     glob.glob(
#         str(base_dir / pattern),
#         recursive=True,
#     )
# )
# batch_size = 100
# batches = [
#     pc_og_files[i : i + batch_size]
#     for i in range(
#         0,
#         len(pc_og_files),
#         batch_size,
#     )
# ]

# # Process each batch and merge results
# final_combined_pcd = None
# for batch_files in batches:
#     batch_pcd = process_batch(batch_files)
#     if final_combined_pcd is None:
#         final_combined_pcd = batch_pcd
#     else:
#         final_combined_pcd += batch_pcd

# # point_clouds = []
# # pbar = tqdm(pc_og_files, unit="files", desc="Step 1: Reading point clouds")
# # for pc in pbar:
# #     pcd = o3d.io.read_point_cloud(pc)
# #     point_clouds.append(pcd)

# # # Step 2: Preprocessing Each Point Cloud
# # # Denoise and estimate normals for each point cloud
# # processed_pcds = []
# # for pcd in tqdm(point_clouds, desc="Step 2: Denoise point clouds"):
# #     # Denoise using Statistical Outlier Removal
# #     pcd, _ = pcd.remove_statistical_outlier(
# #         nb_neighbors=20,
# #         std_ratio=2.0,
# #     )

# #     # Estimate Normals
# #     pcd.estimate_normals(
# #         search_param=o3d.geometry.KDTreeSearchParamHybrid(
# #             radius=0.01,
# #             max_nn=30,
# #         )
# #     )
# #     processed_pcds.append(pcd)

# # # Step 3: Register and Merge Point Clouds
# # # Use pairwise registration to align each point cloud to the first one
# # base_pcd = processed_pcds[0]  # Use the first point cloud as the base
# # transformed_pcds = [base_pcd]

# # for i in tqdm(
# #     range(1, len(processed_pcds)),
# #     desc="Step 3: Transform point clouds",
# # ):
# #     source = processed_pcds[i]
# #     target = base_pcd

# #     # Perform initial alignment using RANSAC
# #     transform = o3d.pipelines.registration.registration_icp(
# #         source,
# #         target,
# #         max_correspondence_distance=0.05,
# #         criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50),
# #     ).transformation

# #     # Apply the transformation to the source point cloud
# #     source.transform(transform)
# #     transformed_pcds.append(source)

# # # Merge all aligned point clouds into one
# # combined_pcd = transformed_pcds[0]
# # for pcd in tqdm(
# #     transformed_pcds[1:],
# #     desc="Merge point clouds",
# # ):
# #     combined_pcd += pcd

# # # Step 4: Refine the Combined Point Cloud
# # # Remove duplicate points
# # combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.005)

# # Save the combined point cloud
# o3d.io.write_point_cloud("final_combined_scene.ply", combined_pcd)

# # Visualize the combined point cloud
# # o3d.visualization.draw_geometries([combined_pcd], window_name="Combined Point Cloud")


import open3d as o3d
import numpy as np
from pathlib import Path
import glob
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_batch(batch_files, voxel_size=0.005):
    try:
        processed_pcds = []

        # Load and preprocess
        for file in tqdm(batch_files, desc="Loading and preprocessing"):
            try:
                pcd = o3d.io.read_point_cloud(file)
                points = np.asarray(pcd.points)

                # Skip if point cloud is empty or too small
                if len(points) < 100:  # Adjust threshold as needed
                    logger.warning(
                        f"Skipping {file}: insufficient points ({len(points)})"
                    )
                    continue

                # Downsample first to reduce processing time
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

                # Check again after downsampling
                if len(np.asarray(pcd.points)) < 100:
                    continue

                # Statistical outlier removal
                pcd, ind = pcd.remove_statistical_outlier(
                    nb_neighbors=20, std_ratio=2.0
                )
                if len(ind) == 0:
                    logger.warning(f"Skipping {file}: no points after outlier removal")
                    continue

                # Estimate normals
                try:
                    pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.01, max_nn=30
                        )
                    )
                except RuntimeError as e:
                    logger.warning(f"Normal estimation failed for {file}: {e}")
                    continue

                processed_pcds.append(pcd)

            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")
                continue

        if not processed_pcds:
            logger.warning("No valid point clouds in batch")
            return None

        # Align PCDs
        base_pcd = processed_pcds[0]
        aligned_pcds = [base_pcd]

        for source in processed_pcds[1:]:
            result = o3d.pipelines.registration.registration_icp(
                source,
                base_pcd,
                0.05,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=50
                ),
            )
            source.transform(result.transformation)
            aligned_pcds.append(source)

        # Combine PCDs
        combined_pcd = aligned_pcds[0]
        for pcd in aligned_pcds[1:]:
            combined_pcd += pcd

        return combined_pcd.voxel_down_sample(voxel_size=voxel_size)

    except Exception as e:
        logger.error(f"Error in process_batch: {e}")
        return None


def main():
    try:
        suffix = "I"
        pattern = f"SyntheticColon_{suffix}/Frames_S1_PC_OG/pc_*.ply"
        base_dir = Path("./datasets/SyntheticColon/")
        pc_og_files = sorted(glob.glob(str(base_dir / pattern)))

        batch_size = 50
        batches = [
            pc_og_files[i : i + batch_size]
            for i in range(0, len(pc_og_files), batch_size)
        ]

        final_pcd = None

        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i+1}/{len(batches)}")
            batch_pcd = process_batch(batch)

            if batch_pcd is None:
                continue

            if final_pcd is None:
                final_pcd = batch_pcd
            else:
                result = o3d.pipelines.registration.registration_icp(
                    batch_pcd,
                    final_pcd,
                    0.05,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=50
                    ),
                )
                batch_pcd.transform(result.transformation)
                final_pcd += batch_pcd
                final_pcd = final_pcd.voxel_down_sample(voxel_size=0.005)

        if final_pcd is not None:
            o3d.io.write_point_cloud("final_combined_scene.ply", final_pcd)
            logger.info("Successfully saved final combined point cloud")

    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
