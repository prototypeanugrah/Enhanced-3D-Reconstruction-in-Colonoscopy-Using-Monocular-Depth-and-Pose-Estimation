import traceback
import numpy as np
import cv2
import vtk
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData

# from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
# from vtkmodules.vtkRenderingAnnotation import vtkAxesActor

# from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
import argparse


def load_camera_intrinsics(
    cam_file: str,
) -> dict:
    """
    Load camera intrinsics from C3VD format

    Args:
        cam_file: Path to camera intrinsics file

    Returns:
        dict: Dictionary containing camera intrinsics
    """
    with open(cam_file, "r") as f:
        params = list(map(float, f.read().strip().split()))

    return {
        "width": int(params[0]),  # 1350
        "height": int(params[1]),  # 1080
        "cx": params[2],  # 679.544839263292
        "cy": params[3],  # 543.975887548343
        "a0": params[4],  # focal length
        # Skip distortion parameters for now
        "e": params[8],  # e
        "f": params[9],  # f
        "g": params[10],  # g
    }


def load_pose(
    pose_file: str,
    frame_idx: int,
) -> np.ndarray:
    """
    Load camera pose for specific frame from pose.txt

    Args:
        pose_file: Path to pose file
        frame_idx: Index of the frame to load pose for

    Returns:
        np.ndarray: 4x4 pose matrix
    """
    with open(pose_file, "r") as f:
        poses = f.readlines()
        pose_line = poses[frame_idx].strip().split(",")
        pose_matrix = np.array([float(x) for x in pose_line]).reshape(4, 4)
    return pose_matrix


def create_point_cloud(
    rgb_img,
    depth_map,
    intrinsics: dict,
    pose: np.ndarray,
    min_depth: float = 0.001,  # 1mm
    max_depth: float = 0.1,  # 100mm
) -> vtkPolyData:
    """
    Generate colored point cloud from RGB and depth images using VTK

    Args:
        rgb_img: RGB image
        depth_map: Depth image
        intrinsics: Camera intrinsics
        pose: Camera pose
        min_depth: Minimum depth value to consider
        max_depth: Maximum depth value to consider

    Returns:
        vtkPolyData: Point cloud as VTK polydata
    """

    height, width = depth_map.shape

    # Create grid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Get parameters
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]
    focal_length = intrinsics["a0"]  # Using a0 as focal length

    # Convert image coordinates to 3D points
    z = depth_map
    x = (x - cx) * z / focal_length
    y = (y - cy) * z / focal_length

    points = np.stack([x, y, z], axis=-1)

    # Filter out invalid depths
    mask = (z > min_depth) & (z < max_depth)
    points = points[mask]
    colors = rgb_img[mask]

    # Transform points to world coordinates
    points_homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1))],
        axis=1,
    )
    points_world = (pose @ points_homogeneous.T).T[:, :3]

    # Create VTK points and polydata
    vtk_points = vtkPoints()
    vtk_colors = vtk.vtkUnsignedCharArray()
    vtk_colors.SetNumberOfComponents(3)
    vtk_colors.SetName("Colors")

    # Add points and colors
    for i in range(len(points_world)):
        vtk_points.InsertNextPoint(points_world[i])
        vtk_colors.InsertNextTuple3(colors[i][0], colors[i][1], colors[i][2])

    # Create the polydata and add points
    polydata = vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.GetPointData().SetScalars(vtk_colors)

    return polydata


def save_point_cloud_visualization(
    polydata: vtkPolyData,
    output_image_path: str,
    width: int = 1024,
    height: int = 768,
) -> None:
    """
    Save point cloud visualization to an image file

    Args:
        polydata: VTK PolyData object
        output_image_path: Path to save the visualization image
        width: Image width
        height: Image height

    Returns:
        None
    """

    try:
        # Create a mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        # Create an actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(5)
        actor.GetProperty().SetRenderPointsAsSpheres(True)

        # Create a renderer
        renderer = vtk.vtkRenderer()
        renderer.AddActor(actor)
        renderer.SetBackground(0.1, 0.1, 0.1)  # Dark gray background

        # Add axes
        axes = vtk.vtkAxesActor()
        # axes.SetTotalLength(0.1, 0.1, 0.1)
        axes.SetTotalLength(0.02, 0.02, 0.02)
        renderer.AddActor(axes)

        # Create render window
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(width, height)
        render_window.SetOffScreenRendering(1)  # Enable off-screen rendering

        # Reset camera and adjust view
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()

        # Get point cloud center for better camera positioning
        center = polydata.GetCenter()
        bounds = polydata.GetBounds()
        diagonal = np.sqrt(
            (bounds[1] - bounds[0]) ** 2
            + (bounds[3] - bounds[2]) ** 2
            + (bounds[5] - bounds[4]) ** 2
        )

        # Render from multiple angles
        angles = [0, 45, 90, 180]  # Different viewing angles
        for i, angle in enumerate(angles):

            # Position camera relative to point cloud center
            distance = diagonal * 2  # Camera distance from center
            camera.SetPosition(center[0], center[1], center[2] + distance)
            camera.SetFocalPoint(center)

            # Reset camera position for each angle
            # camera.SetPosition(0, 0, 1)  # Set camera position
            # camera.SetFocalPoint(0, 0, 0)  # Look at origin
            camera.SetViewUp(0, 1, 0)  # Set up direction

            # Rotate camera
            # camera = renderer.GetActiveCamera()
            camera.Azimuth(angle)
            camera.Elevation(30)

            # Render
            renderer.ResetCamera()
            camera.Zoom(1.8)
            render_window.Render()

            # Set up window to image filter
            window_to_image = vtk.vtkWindowToImageFilter()
            window_to_image.SetInput(render_window)
            window_to_image.Update()

            # Save to file
            writer = vtk.vtkPNGWriter()
            output_path = f"{output_image_path}_view{angle}.png"
            writer.SetFileName(output_path)
            writer.SetInputConnection(window_to_image.GetOutputPort())
            writer.Write()

    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        traceback.print_exc()  # Print full error traceback

    finally:
        # Clean up
        if "render_window" in locals():
            render_window.Finalize()


def visualize_point_cloud(polydata: vtkPolyData) -> None:
    """
    Visualize the point cloud using VTK

    Args:
        polydata: VTK PolyData object

    Returns:
        None
    """
    # Create a mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    # Create an actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(2)  # Make points more visible

    # Create a renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.1, 0.1)  # Dark gray background

    # Create axes actor
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(0.1, 0.1, 0.1)  # Set axes length to 10cm
    axes.SetShaftTypeToLine()
    renderer.AddActor(axes)

    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1024, 768)  # Window size

    # Create an interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # Set the interactor style
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

    # Initialize and start
    interactor.Initialize()
    render_window.Render()

    # Add orientation widget
    axes_widget = vtk.vtkOrientationMarkerWidget()
    axes_widget.SetOrientationMarker(axes)
    axes_widget.SetInteractor(interactor)
    axes_widget.EnabledOn()
    axes_widget.InteractiveOn()

    print("\nVisualization Controls:")
    print("- Left mouse button: Rotate")
    print("- Middle mouse button: Pan")
    print("- Right mouse button: Zoom")
    print("- 'r' key: Reset camera")
    print("Press 'q' to close the window")

    interactor.Start()


def save_vtk_point_cloud(
    polydata: vtkPolyData,
    output_path: str,
) -> None:
    """
    Save point cloud to file based on extension

    Args:
        polydata: VTK PolyData object
        output_path: Path to save the point cloud

    Returns:
        None
    """

    # Add verification
    n_points = polydata.GetNumberOfPoints()
    print(f"Saving point cloud with {n_points} points")

    if n_points == 0:
        print("Warning: Point cloud is empty!")
        return

    # Get file extension
    extension = output_path.lower().split(".")[-1]

    if extension == "ply":
        # PLY writer
        writer = vtk.vtkPLYWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(polydata)
        writer.SetArrayName("Colors")  # Specify the color array name
        writer.SetColorMode(1)  # Use point colors
        writer.SetComponent(0)  # RGB color components
        writer.SetFileTypeToBinary()  # Can use SetFileTypeToASCII() for text format
        writer.Write()

    elif extension == "vtp":
        # VTK XML PolyData writer
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(polydata)
        writer.SetCompressorTypeToZLib()  # Optional compression
        writer.SetDataModeToBinary()  # Binary format for smaller file size
        writer.Write()

    else:
        raise ValueError(f"Unsupported file extension: {extension}. Use .ply or .vtp")


def analyze_point_cloud_distribution(
    points_world: np.ndarray,
    colors: np.ndarray,
) -> None:
    """
    Analyze the distribution of points and colors in the point cloud

    Args:
        points_world: 3D points in world coordinates
        colors: RGB colors for each point

    Returns:
        None
    """
    print("\nDetailed Point Cloud Analysis:")

    # Point distribution
    print("\nPoint Distribution (meters):")
    for axis, name in enumerate(["X", "Y", "Z"]):
        values = points_world[:, axis]
        print(f"{name} axis:")
        print(f"  Range: {values.min():.3f}m to {values.max():.3f}m")
        print(f"  Mean: {values.mean():.3f}m")
        print(f"  Std Dev: {values.std():.3f}m")

    # Color distribution
    print("\nColor Distribution:")
    for i, color in enumerate(["Red", "Green", "Blue"]):
        values = colors[:, i]
        print(f"{color}:")
        print(f"  Range: {values.min()} to {values.max()}")
        print(f"  Mean: {values.mean():.1f}")
        print(f"  Std Dev: {values.std():.1f}")

    # Point density
    volume = np.prod(np.ptp(points_world, axis=0))
    density = len(points_world) / volume if volume > 0 else 0
    print(f"\nPoint Density: {density:.1f} points/m³")


def process_sequence(
    rgb_path: str,
    depth_path: str,
    cam_file: str,
    pose_file: str,
    output_path: str,
    visualize: bool = False,
    save_vis: bool = False,
) -> vtkPolyData:
    """
    Process a sequence of frames to generate point clouds

    Args:
        rgb_path: Path to RGB image
        depth_path: Path to depth image
        cam_file: Path to camera intrinsics file
        pose_file: Path to pose file
        output_path: Path to save output point cloud
        visualize: Visualize the point cloud
        save_vis: Save visualization images

    Returns:
        vtkPolyData: Point cloud as VTK polydata
    """
    # Load camera parameters
    intrinsics = load_camera_intrinsics(cam_file)

    # Load RGB and depth images
    rgb = cv2.imread(rgb_path)
    # print(f"RGB image shape: {rgb.shape}")
    # print(f"RGB image range: {rgb.min()} to {rgb.max()}")

    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    # Scale depth values to the correct range (0-100mm = 0-0.1m)
    depth = (depth / 65535.0) * 0.1  # Scale to range 0-0.1m
    # print(f"Depth image shape: {depth.shape}")
    # print(f"Depth image range: {depth.min()} to {depth.max()}")
    # print(f"Depth range in meters: {depth.min():.3f}m to {depth.max():.3f}m")

    # Get frame index from filename
    frame_idx = int(rgb_path.split("/")[-1].split("_")[0])

    # Load pose for this frame
    pose = load_pose(pose_file, frame_idx)

    # Create point cloud
    polydata = create_point_cloud(rgb, depth, intrinsics, pose)

    # # Get points and colors for analysis
    # points = np.array(
    #     [polydata.GetPoint(i) for i in range(polydata.GetNumberOfPoints())]
    # )
    # colors = np.array(
    #     [
    #         polydata.GetPointData().GetScalars().GetTuple3(i)
    #         for i in range(polydata.GetNumberOfPoints())
    #     ]
    # )
    # analyze_point_cloud_distribution(points, colors)

    # Save point cloud (optional - you'll need to implement this if needed)
    save_vtk_point_cloud(polydata, output_path)

    # Save visualization images
    if save_vis:
        visualization_path = output_path.rsplit(".", 1)[0]  # Remove .ply extension
        save_point_cloud_visualization(polydata, visualization_path)

    # Visualize if requested
    if visualize:
        visualize_point_cloud(polydata)

    return polydata


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
        "-c",
        "--cam_file",
        help="Path to camera.txt file",
        required=False,
        default="datasets/C3VD/calibration/cam.txt",
    )
    parser.add_argument(
        "-p",
        "--pose_file",
        help="Path to pose.txt file",
        required=False,
        default="datasets/C3VD/trans_t1_a/pose.txt",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Path to save output point cloud",
        required=True,
    )
    # parser.add_argument(
    #     "--vis",
    #     help="Visualize the point cloud",
    #     default=False,
    # )
    args = parser.parse_args()

    # # Hardcoded paths for camera and pose files
    # cam_file = "datasets/C3VD/calibration/cam.txt"
    # pose_file = "datasets/C3VD/trans_t1_a/pose.txt"

    pcd = process_sequence(
        args.rgb_path,
        args.depth_path,
        args.cam_file,
        args.pose_file,
        args.output_path,
        # args.vis,
        save_vis=False,
    )

# Usage: python point_cloud.py -r datasets/C3VD/trans_t1_a/0000_color.png -d datasets/C3VD/trans_t1_a/0000_depth.tiff -o pc_trans_t1_a_0000.ply
