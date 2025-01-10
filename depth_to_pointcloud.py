"""
Script to generate point clouds from depth and color images using Open3D.
The script processes a directory of images and generates a unified point cloud
and mesh using the Poisson reconstruction method.

The script assumes the following directory structure:
- SyntheticColon
    - SyntheticColon_1
        - Frames_S1 # Folder containing original RGB images and depth images
            - FrameBuffer_*.png
            - Depth_*.png
        - Frames_S1_OP # Folder containing predicted depth images
        - Frames_S2
        - SavedPosition_S1.txt
        - SavedRotationQuaternion_S1.txt
        - cam.txt
    - SyntheticColon_2
    - SyntheticColon_3
    
The script can be run with the following command:
python depth_to_pointcloud_gpt.py \
    -i <path_to_images> \
    -d <path_to_depth_images> \
    -o <output_dir> \
    -ds <dataset_type>

Arguments:
- img-path: Path to the input image or directory containing images.
- depth-path: Path to the depth image or directory containing depth images.
- outdir: Directory to save the output point clouds.
- ds_type: Type of dataset to run the script for. Options: simcol, testing, c3vd

Returns:
- combined_point_cloud.ply: Combined point cloud of all frames.
- combined_mesh.ply: Mesh generated from the combined point cloud.

Note: The script requres camera intrinsics, position and rotation files for 
each procedure.
"""

import os
import argparse
import glob

from pathlib import Path
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import cv2
import numpy as np
import open3d as o3d


def input_output_files(args: argparse.Namespace) -> tuple:
    """
    Get the list of image and depth files to process.

    Args:
        args: Command-line arguments.

    Returns:
        Tuple: List of RGB, depth image files, and output directory.
    """
    # Get the list of image files to process
    rgb_filenames = []
    depth_filenames = []
    if os.path.isfile(args.img_path):
        if args.img_path.endswith("txt"):
            with open(args.img_path, "r", encoding="utf-8") as f:
                rgb_filenames = f.read().splitlines()
        if args.depth_path.endswith("txt"):
            with open(args.depth_path, "r", encoding="utf-8") as f:
                depth_filenames = f.read().splitlines()
        else:
            # Single image processing
            rgb_filenames = [args.img_path]
            if args.outdir is None:
                args.outdir = str(Path(args.img_path).parent)
    elif args.ds_type == "simcol":
        # SimCol dataset processing
        base_dir = Path(args.img_path)
        for suffix in ["I", "II", "III"]:
            rgb_pattern = f"SyntheticColon_{suffix}/Frames_*/FrameBuffer_*.png"
            depth_pattern = f"SyntheticColon_{suffix}/Frames_*_OP/depth/Depth_*.png"
            paths = [
                p
                for p in glob.glob(
                    str(base_dir / rgb_pattern),
                    recursive=True,
                )
                if "_OP" not in str(p)
            ]
            rgb_filenames.extend(sorted(paths))
            depth_filenames.extend(
                sorted(
                    glob.glob(
                        str(base_dir / depth_pattern),
                        recursive=True,
                    )
                )
            )
        if args.outdir is None:
            args.outdir = str(base_dir)

    elif args.ds_type == "testing":
        base_dir = Path(args.img_path)
        pattern = "frame_*.jpg"
        rgb_filenames.extend(
            sorted(
                glob.glob(
                    str(base_dir / pattern),
                    recursive=True,
                )
            )
        )
        if args.outdir is None:
            args.outdir = str(base_dir)

    return (
        rgb_filenames,
        depth_filenames,
        args.outdir,
    )


# ---------- Utility Functions ----------
def load_camera_intrinsics(
    file_path: str,
    width: int,
    height: int,
) -> o3d.camera.PinholeCameraIntrinsic:
    """
    Loads camera intrinsics from a text file with fx, fy, cx, cy values.

    Args:
        file_path: Path to the text file containing camera intrinsics.

    Returns:
        o3d.camera.PinholeCameraIntrinsic: Camera intrinsics object
    """
    intrinsics_values = np.loadtxt(file_path).reshape(3, 3)
    fx, fy = intrinsics_values[0, 0], intrinsics_values[1, 1]
    cx, cy = intrinsics_values[0, 2], intrinsics_values[1, 2]

    return o3d.camera.PinholeCameraIntrinsic(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def load_transformation(
    position_file,
    rotation_file,
    frame_idx,
):
    """Loads position and rotation quaternion from text files and converts to a 4x4 transformation matrix."""
    position = np.loadtxt(position_file)
    quaternion = np.loadtxt(rotation_file)

    # Extract position and quaternion for the given frame
    position_frame = position[frame_idx]
    quaternion_frame = quaternion[frame_idx]

    # Convert quaternion to rotation matrix using scipy
    rotation_matrix = R.from_quat(quaternion_frame).as_matrix()

    # Create transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = rotation_matrix
    transformation[:3, 3] = position_frame
    return transformation


# ---------- Point Cloud Generation ----------
def generate_point_cloud(
    depth_image_path: str,
    color_image_path: str,
    intrinsics_path: str,
    position_file: str,
    rotation_file: str,
    frame_idx: int,
) -> o3d.geometry.PointCloud:
    """
    Generates a point cloud from a depth image and applies the camera intrinsics and transformation.

    Args:
        depth_image_path(str): Path to the depth image.
        color_image_path(str): Path to the color image.
        intrinsics_path(str): Path to the camera intrinsics file.
        position_file(str): Path to the position file.
        rotation_file(str): Path to the rotation file.
        frame_idx(int): Index of the frame in the procedure.

    Returns:
        o3d.geometry.PointCloud: Point cloud generated from the depth image.
    """

    # Load depth and color images
    depth_image = cv2.imread(
        depth_image_path, cv2.IMREAD_UNCHANGED
    )  # Assuming 16-bit depth image
    color_image = cv2.imread(color_image_path)
    width, height = color_image.shape[:2]
    depth_image = cv2.resize(
        depth_image,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    # Convert images to Open3D formats
    depth = o3d.geometry.Image(depth_image)
    color = o3d.geometry.Image(color_image)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        convert_rgb_to_intensity=False,
    )

    # Load intrinsics
    intrinsics = load_camera_intrinsics(intrinsics_path, width, height)

    # Create point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics,
    )

    # Load transformation and apply to point cloud
    transformation = load_transformation(
        position_file,
        rotation_file,
        frame_idx,
    )
    point_cloud.transform(transformation)

    return point_cloud


# ---------- Mesh Generation ----------
def generate_mesh(
    point_cloud: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh:
    """
    Generates a 3D mesh from a point cloud using the Poisson reconstruction

    Args:
        point_cloud(o3d.geometry.PointCloud): Input point cloud.

    Returns:
        o3d.geometry.TriangleMesh: Mesh generated from the point cloud.
    """
    # Downsample the point cloud
    point_cloud = point_cloud.voxel_down_sample(voxel_size=0.01)

    # Estimate normals
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1,  # Radius for normal estimation
            max_nn=30,  # Maximum number of nearest neighbors
        )
    )

    # Apply Poisson reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud,  # Input point cloud
        depth=9,  # Depth of the octree (Means: 2^depth)
    )

    # Optionally remove low-density vertices
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.percentile(
        densities, 5
    )  # Remove 5% of the vertices
    mesh.remove_vertices_by_mask(vertices_to_remove)  # Remove vertices with low density

    return mesh


def get_procedure_files(rgb_filename: str) -> tuple:
    """
    Get corresponding camera, position and rotation files for a procedure.

    Args:
        rgb_filename(str): Path to the RGB image.

    Returns:
        Tuple: Paths to camera, position and rotation files.
    """
    # Extract procedure path from rgb filename
    # Example: .../SyntheticColon_I/Frames_S1/FrameBuffer_*.png
    path = Path(rgb_filename)
    procedure_dir = path.parent.parent  # Go up two levels to reach SyntheticColon_X
    procedure_name = path.parent.name  # Get Frames_SX
    subprocedure_name = procedure_name.split("_")[1]

    # Get camera file (same for all frames in a SyntheticColon_X directory)
    cam_file = procedure_dir / "cam.txt"

    # Get position and rotation files (specific to each procedure Frames_SX)
    position_file = procedure_dir / f"SavedPosition_{subprocedure_name}.txt"
    rotation_file = procedure_dir / f"SavedRotationQuaternion_{subprocedure_name}.txt"

    return (
        str(cam_file),
        str(position_file),
        str(rotation_file),
    )


# ---------- Main Processing ----------
def main(
    depth_image_paths: list,
    color_image_paths: list,
    output_dir: str,
) -> None:
    """
    Processes a procedure to generate a unified point cloud and mesh.

    Args:
        depth_image_paths(list): List of paths to depth images.
        color_image_paths(list): List of paths to color images.
        output_dir(str): Output directory to save the point cloud and mesh.

    Returns:
        None
    """
    combined_point_cloud = o3d.geometry.PointCloud()

    for frame_idx, (depth_image_path, color_image_path) in tqdm(
        enumerate(zip(depth_image_paths, color_image_paths)),
        desc="Processing frames",
        total=len(color_image_paths),
    ):

        # Get corresponding camera, position and rotation files
        intrinsics_path, position_file, rotation_file = get_procedure_files(
            color_image_path
        )

        # Generate point cloud for each frame
        point_cloud = generate_point_cloud(
            depth_image_path,
            color_image_path,
            intrinsics_path,
            position_file,
            rotation_file,
            frame_idx,
        )
        combined_point_cloud += point_cloud

    # Merge and downsample the combined point cloud
    combined_point_cloud = combined_point_cloud.voxel_down_sample(
        voxel_size=0.01,
    )

    # Generate mesh from the combined point cloud
    mesh = generate_mesh(combined_point_cloud)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save point cloud and mesh
    o3d.io.write_point_cloud(
        f"{output_dir}/combined_point_cloud.ply", combined_point_cloud
    )
    o3d.io.write_triangle_mesh(f"{output_dir}/combined_mesh.ply", mesh)

    # # Visualize (optional)
    # o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate point clouds from images.",
    )
    parser.add_argument(
        "-i",
        "--img-path",
        type=str,
        required=True,
        help="Path to the input image or directory containing images.",
    )
    parser.add_argument(
        "-d",
        "--depth-path",
        type=str,
        help="Path to the depth image or directory containing depth images.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        # default="./vis_pointcloud",
        help="Directory to save the output point clouds.",
    )
    parser.add_argument(
        "-ds",
        "--ds_type",
        type=str,
        choices=["simcol", "testing", "c3vd"],
        help="Type of dataset to run the script for",
    )

    args = parser.parse_args()

    # Get the list of image and depth files to process
    rgb_filenames, depth_filenames, outdir = input_output_files(args)

    # Process the procedures
    main(
        depth_filenames,
        rgb_filenames,
        outdir,
    )
