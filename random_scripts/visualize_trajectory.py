import open3d as o3d
import numpy as np
from pathlib import Path
import glob
import argparse


def load_camera_poses(position_file: Path, quaternion_file: Path) -> np.ndarray:
    """Load camera positions and orientations"""
    positions = np.loadtxt(str(position_file))
    quaternions = np.loadtxt(str(quaternion_file))

    poses = []
    for pos, quat in zip(positions, quaternions):
        # Convert quaternion to right-handed coordinate system
        quat_right_handed = [quat[0], -quat[1], -quat[2], quat[3]]
        rot_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quat_right_handed)

        # Create 4x4 transformation matrix
        pose = np.eye(4)
        pose[:3, :3] = rot_matrix
        pose[:3, 3] = pos
        poses.append(pose)

    return poses


def create_camera_markers(poses, size=0.02):
    """Create camera frustum markers for visualization"""
    camera_markers = []

    for pose in poses:
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        mesh.transform(pose)
        camera_markers.append(mesh)

    return camera_markers


def create_trajectory_line(poses, color=[1, 0, 0]):
    """Create a line connecting camera positions"""
    points = [pose[:3, 3] for pose in poses]
    lines = [[i, i + 1] for i in range(len(points) - 1)]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in lines])

    return line_set


def visualize_trajectory(base_dir: Path, sequence: str):
    """Visualize point clouds and camera trajectory"""
    # Load point clouds
    pc_pattern = f"{sequence}_PC/pc_*.ply"
    pc_files = sorted(glob.glob(str(base_dir / pc_pattern)))

    # Load camera poses
    sequence_num = sequence.split("_")[1]
    position_file = base_dir / f"SavedPosition_{sequence_num}.txt"
    quaternion_file = base_dir / f"SavedRotationQuaternion_{sequence_num}.txt"
    poses = load_camera_poses(position_file, quaternion_file)

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add point clouds with different colors
    print("Loading point clouds...")
    for i, pc_file in enumerate(pc_files):
        pcd = o3d.io.read_point_cloud(pc_file)
        # Optional: downsample to improve performance
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        vis.add_geometry(pcd)

    # Add camera markers and trajectory
    print("Adding camera trajectory...")
    camera_markers = create_camera_markers(poses)
    for marker in camera_markers:
        vis.add_geometry(marker)

    trajectory_line = create_trajectory_line(poses)
    vis.add_geometry(trajectory_line)

    # Set visualization options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # Black background
    opt.point_size = 1.0

    # Set initial viewpoint
    ctr = vis.get_view_control()
    ctr.set_zoom(0.1)

    print("Visualization ready. Press 'Q' to exit.")
    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SimCol3D trajectory and point clouds"
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        help="Base directory (e.g., SyntheticColon_I)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--sequence",
        help="Sequence name (e.g., Frames_S1)",
        type=str,
        required=True,
    )
    # # Additional arguments you might want to add to the parser:
    # parser.add_argument(
    #     "--downsample",
    #     type=float,
    #     default=0.01,
    #     help="Voxel size for downsampling (default: 0.01)",
    # )
    # parser.add_argument(
    #     "--camera-size",
    #     type=float,
    #     default=0.02,
    #     help="Size of camera markers (default: 0.02)",
    # )
    # parser.add_argument(
    #     "--skip-frames",
    #     type=int,
    #     default=1,
    #     help="Process every nth frame (default: 1)",
    # )
    args = parser.parse_args()

    visualize_trajectory(args.input_dir, args.sequence)


if __name__ == "__main__":
    main()
