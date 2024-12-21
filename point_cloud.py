import argparse
import os

from scipy.spatial.transform import Rotation
import cv2
import numpy as np
import open3d as o3d


def load_camera_intrinsics(
    cam_file: str,
    dataset_type: str = "c3vd",
) -> dict:
    """
    Load camera intrinsics from C3VD or SimCol3D format

    Args:
        cam_file: Path to camera intrinsics file
        dataset_type: Type of dataset ("c3vd" or "simcol3d")

    Returns:
        dict: Dictionary containing camera intrinsics
    """
    with open(cam_file, "r") as f:
        params = list(map(float, f.read().strip().split()))

    if dataset_type.lower() == "c3vd":
        return {
            "width": int(params[0]),
            "height": int(params[1]),
            "cx": params[2],
            "cy": params[3],
            "a0": params[4],
            "e": params[8],
            "f": params[9],
            "g": params[10],
        }
    else:  # simcol3d
        return {
            "width": 475,  # Fixed size for SimCol3D
            "height": 475,
            "fx": params[0],
            "fy": params[4],
            "cx": params[2],
            "cy": params[5],
        }


def load_pose(
    pose_file: str,
    frame_idx: int,
    dataset_type: str = "c3vd",
    position_file: str = None,
    quaternion_file: str = None,
) -> np.ndarray:
    """
    Load camera pose for specific frame

    Args:
        pose_file: Path to pose file (C3VD)
        frame_idx: Index of the frame
        dataset_type: Type of dataset ("c3vd" or "simcol3d")
        position_file: Path to position file (SimCol3D)
        quaternion_file: Path to quaternion file (SimCol3D)

    Returns:
        np.ndarray: 4x4 pose matrix
    """
    if dataset_type.lower() == "c3vd":
        with open(pose_file, "r") as f:
            poses = f.readlines()
            pose_line = poses[frame_idx].strip().split(",")
            pose_matrix = np.array([float(x) for x in pose_line]).reshape(4, 4)
        return pose_matrix

    else:  # simcol3d
        # Load position and quaternion
        positions = np.loadtxt(position_file)
        quaternions = np.loadtxt(quaternion_file)

        # Get position and quaternion for specific frame
        position = positions[frame_idx]
        quat = quaternions[frame_idx]

        # Convert quaternion to rotation matrix (adjust for left-handed coordinate system)
        quat_right_handed = [
            quat[0],
            -quat[1],
            -quat[2],
            quat[3],
        ]  # Convert to right-handed

        rot_matrix = Rotation.from_quat(quat_right_handed).as_matrix()

        # Create 4x4 transformation matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rot_matrix
        pose_matrix[:3, 3] = position

        return pose_matrix


def create_point_cloud(
    rgb_img,
    depth_map,
    intrinsics: dict,
    pose: np.ndarray,
    min_depth: float = 0.001,
    max_depth: float = 0.1,
    dataset_type: str = "c3vd",
) -> o3d.geometry.PointCloud:
    """
    Generate colored point cloud from RGB and depth images using Open3D

    Args:
        rgb_img: RGB image
        depth_map: Depth image
        intrinsics: Camera intrinsics
        pose: Camera pose
        min_depth: Minimum depth value
        max_depth: Maximum depth value
        dataset_type: Type of dataset ("c3vd" or "simcol3d")

    Returns:
        o3d.geometry.PointCloud: Colored point cloud
    """
    if len(rgb_img.shape) == 3:
        height, width, _ = rgb_img.shape
    else:
        height, width = rgb_img.shape

    # Resize depth map to match RGB dimensions
    if rgb_img.shape[:2] != depth_map.shape[:2]:
        # tqdm.write(f"Resizing depth from {depth_map.shape} to {(height, width)}")
        depth_map = cv2.resize(
            depth_map,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    if dataset_type.lower() == "c3vd":
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        focal_length = intrinsics["a0"]
        min_depth = 0.001  # 1mm
        max_depth = 0.1  # 100mm
    else:  # simcol3d
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        focal_length = intrinsics["fx"]  # Using fx as focal length
        min_depth = 0.0  # 1mm
        max_depth = 20.0  # 20cm

    # Convert image coordinates to 3D points
    z = depth_map
    x = (x - cx) * z / focal_length
    y = (y - cy) * z / focal_length

    points = np.stack([x, y, z], axis=-1)

    # Filter out invalid depths
    mask = (z > min_depth) & (z < max_depth)
    points = points[mask]
    colors = rgb_img[mask]

    if dataset_type == "simcol3d":
        # Convert points to meters before applying transformation
        points = points / 100.0  # Convert from cm to meters

    # Transform points to world coordinates
    points_homogeneous = np.concatenate(
        [
            points,
            np.ones((points.shape[0], 1)),
        ],
        axis=1,
    )
    points_world = (pose @ points_homogeneous.T).T[:, :3]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    return pcd


def process_sequence(
    rgb_path: str,
    depth_path: str,
    cam_file: str,
    pose_file: str,
    output_path: str,
    dataset_type: str = "c3vd",
    position_file: str = None,
    quaternion_file: str = None,
) -> o3d.geometry.PointCloud:
    """
    Process a sequence of frames to generate point clouds

    Args:
        rgb_path: Path to RGB image
        depth_path: Path to depth image
        cam_file: Path to camera intrinsics file
        pose_file: Path to pose file (C3VD)
        output_path: Path to save output point cloud
        dataset_type: Type of dataset ("c3vd" or "simcol3d")
        position_file: Path to position file (SimCol3D)
        quaternion_file: Path to quaternion file (SimCol3D)
        visualize: Visualize the point cloud
        save_vis: Save visualization images
    """
    # Load camera parameters
    intrinsics = load_camera_intrinsics(cam_file, dataset_type)

    # Load RGB and depth images
    rgb = cv2.imread(rgb_path)
    # print(f"RGB image shape: {rgb.shape}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    # depth = (depth / 65535.0) * 0.1  # Scale to range 0-0.1m

    # print(f"Depth image shape: {depth.shape}")
    # print(f"Depth range before scaling: {depth.min()} to {depth.max()}")

    if dataset_type.lower() == "c3vd":
        depth = (depth / 65535.0) * 0.1  # Scale to range 0-0.1m
    else:  # simcol3d
        # Convert depth from [0,1] to [0,20] cm
        depth = depth.astype(float)
        if depth.max() <= 1.0:  # Check if values are in [0,1] range
            depth = depth * 20.0  # Scale to [0,20] cm
        else:
            depth = depth / 255.0 * 20.0  # If values are in [0,255] range

    # print(f"Depth range after scaling: {depth.min():.6f} to {depth.max():.6f}")

    # Get frame index from filename
    if dataset_type.lower() == "c3vd":
        frame_idx = int(rgb_path.split("/")[-1].split("_")[0])
    else:  # simcol3d
        # Extract number from "FrameBuffer_XXXX.png"
        frame_idx = int(rgb_path.split("FrameBuffer_")[-1].split(".")[0])

    # Load pose for this frame
    pose = load_pose(
        pose_file,
        frame_idx,
        dataset_type,
        position_file,
        quaternion_file,
    )

    # Create point cloud
    pcd = create_point_cloud(
        rgb,
        depth,
        intrinsics,
        pose,
        dataset_type=dataset_type,
    )

    # Add verification
    n_points = len(pcd.points)
    # print(f"Saving point cloud with {n_points} points")

    if n_points == 0:
        print("Warning: Point cloud is empty!")
        return

    o3d.io.write_point_cloud(os.path.join(output_path + ".ply"), pcd)

    return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate point cloud from RGB and depth images"
    )
    parser.add_argument(
        "-r",
        "--rgb_path",
        help="Path to RGB image",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--depth_path",
        help="Path to depth image",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--dataset_type",
        help="Dataset type (c3vd or simcol3d)",
        choices=["c3vd", "simcol3d"],
        required=True,
    )
    parser.add_argument(
        "-c",
        "--cam_file",
        help="Path to camera.txt file",
        required=True,
        # default="datasets/C3VD/calibration/cam.txt",
    )
    parser.add_argument(
        "-p",
        "--pose_file",
        help="Path to pose.txt file (C3VD only)",
        required=False,
        default="datasets/C3VD/trans_t1_a/pose.txt",
    )
    parser.add_argument(
        "-sp",
        "--position_file",
        help="Path to SavedPosition file (SimCol3D only)",
        required=False,
    )
    parser.add_argument(
        "-q",
        "--quaternion_file",
        help="Path to SavedRotationQuaternion file (SimCol3D only)",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Path to save output point cloud",
        required=True,
    )
    args = parser.parse_args()

    pcd = process_sequence(
        args.rgb_path,
        args.depth_path,
        args.cam_file,
        args.pose_file,
        args.output_path,
        args.dataset_type,
        args.position_file,
        args.quaternion_file,
    )

# Usage:
# C3VD: python point_cloud.py -r datasets/C3VD/trans_t1_a/0000_color.png -d datasets/C3VD/trans_t1_a/0000_depth.tiff -o pc_open_trans_t1_a_0000.ply -t c3vd -c datasets/C3VD/calibration/cam.txt
# SimCol3d: python point_cloud.py -r datasets/SyntheticColon/SyntheticColon_I/Frames_S1/FrameBuffer_0000.png -d datasets/SyntheticColon/SyntheticColon_I/Frames_S1_OP/depth/FrameBuffer_0000.png -c datasets/SyntheticColon/SyntheticColon_I/cam.txt -sp datasets/SyntheticColon/SyntheticColon_I/SavedPosition_S1.txt -t simcol3d -q datasets/SyntheticColon/SyntheticColon_I/SavedRotationQuaternion_S1.txt -o pc_open_simcol_s1_0000.ply
