import os
from pathlib import Path


def count_color_images(base_path):
    # Convert string path to Path object if needed
    base_path = Path(base_path)

    # Dictionary to store counts for each folder
    folder_counts = {}

    # Iterate through all folders in C3VD directory
    for folder_path in base_path.iterdir():
        if folder_path.is_dir():
            # Count files ending with _color.png in this folder
            color_images = list(folder_path.glob("*_color.png"))
            folder_counts[folder_path.name] = len(color_images)

    return folder_counts


# def count_framebuffer_images(base_path):
#     # Convert string path to Path object if needed
#     base_path = Path(base_path)

#     # Print to verify the path exists
#     # print(f"Checking path: {base_path}")
#     # print(f"Path exists: {base_path.exists()}")

#     # Dictionary to store counts for each folder
#     folder_counts = {}

#     # Iterate through SyntheticColon folders
#     for colon_folder in base_path.iterdir():
#         # print(f"Found folder: {colon_folder}")  # Debug print
#         if colon_folder.is_dir():
#             for sequence_folder in colon_folder.iterdir():
#                 # print(f"Sequence folder: {sequence_folder}")
#                 if sequence_folder.is_dir():
#                     for frames_folder in sequence_folder.iterdir():
#                         if (
#                             frames_folder.is_dir()
#                             and frames_folder.name.startswith("Frames_")
#                             and not frames_folder.name.endswith("_OP")
#                         ):
#                             print(f"Counting in: {frames_folder}")
#                             # Count files starting with FrameBuffer_ in this folder
#                             framebuffer_images = list(
#                                 frames_folder.glob("FrameBuffer_*.png")
#                             )
#                             folder_counts[
#                                 f"{colon_folder.name}/{sequence_folder.name}/{frames_folder.name}"
#                             ] = len(framebuffer_images)

#     return folder_counts


def count_framebuffer_images(base_path):
    base_path = Path(base_path)
    print(f"Checking path: {base_path}")

    folder_counts = {}

    # Iterate through SyntheticColon folders
    for colon_folder in base_path.iterdir():
        if colon_folder.is_dir() and colon_folder.name.startswith("SyntheticColon_"):
            print(f"Processing colon folder: {colon_folder.name}")

            # Find all Frames_* folders that don't end with _OP
            frames_folders = [
                f
                for f in colon_folder.iterdir()
                if f.is_dir()
                and f.name.startswith("Frames_")
                and not f.name.endswith("_OP")
            ]

            for frames_folder in frames_folders:
                print(f"Counting FrameBuffer images in: {frames_folder.name}")
                framebuffer_images = list(frames_folder.glob("FrameBuffer_*.png"))
                if framebuffer_images:
                    folder_counts[f"{colon_folder.name}/{frames_folder.name}"] = len(
                        framebuffer_images
                    )
                print(f"Found {len(framebuffer_images)} images")

    return folder_counts


if __name__ == "__main__":
    # Adjust this path to point to your C3VD folder
    c3vd_path = "./datasets/SyntheticColon/"

    # counts = count_color_images(c3vd_path)
    counts = count_framebuffer_images(c3vd_path)

    # # Print results
    # print("Color image counts by folder:")
    # for folder, count in sorted(counts.items()):
    #     print(f"{folder}: {count} color images")

    print("\nFrameBuffer image counts by folder:")
    if counts:
        for folder, count in sorted(counts.items()):
            print(f"{folder}: {count} images")
    else:
        print("No FrameBuffer images found!")
