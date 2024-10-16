import os
import shutil

from sklearn.model_selection import train_test_split


def split_and_move_videos(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Split videos into train, validation, and test sets and move them to
    respective directories.

    Args:
        input_dir: The directory containing the video files.
        output_dir: The directory where the split videos will be saved.
        train_ratio: The ratio of the dataset to use for training. Default is 0.7.
        val_ratio: The ratio of the dataset to use for validation. Default is 0.15.

    Returns:
        train_dir: The directory containing the training videos.
        val_dir: The directory containing the validation videos.
        test_dir: The directory containing the test videos.
    """
    # Get all video files
    video_files = [f for f in os.listdir(input_dir) if f.endswith((".mp4", ".avi"))]

    # Split the dataset
    train_val, test = train_test_split(
        video_files,
        test_size=1 - train_ratio - val_ratio,
        random_state=42,
    )
    train, val = train_test_split(
        train_val,
        test_size=val_ratio / (train_ratio + val_ratio),
        random_state=42,
    )

    # Create directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")

    for directory in [train_dir, val_dir, test_dir]:
        os.makedirs(directory, exist_ok=True)

    # Move files
    for video in train:
        shutil.move(os.path.join(input_dir, video), os.path.join(train_dir, video))
    for video in val:
        shutil.move(os.path.join(input_dir, video), os.path.join(val_dir, video))
    for video in test:
        shutil.move(os.path.join(input_dir, video), os.path.join(test_dir, video))

    return train_dir, val_dir, test_dir


if __name__ == "__main__":
    input_dir = "hyper-kvasir-videos/videos"
    output_dir = "datasets/hyper-kvasir"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dir, val_dir, test_dir = split_and_move_videos(input_dir, output_dir)
    print(f"Training videos saved in: {train_dir}")
    print(f"Validation videos saved in: {val_dir}")
    print(f"Test videos saved in: {test_dir}")
