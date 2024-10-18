"Module for custom dataset"

import os

from torch.utils.data import Dataset
from torchvision import transforms

import cv2


class CustomVideoDepthDataset(Dataset):
    """
    Custom dataset class for loading video frames for depth estimation.

    Args:
        video_dir (str): The directory containing the video files.
        transform (callable, optional): Optional transform to be applied to
        each video frame.
    """

    def __init__(
        self,
        video_dir,
        transform=None,
        single_file=None,
    ):
        self.video_dir = video_dir
        self.transform = transform
        self.single_file = single_file
        self.frames = self._load_frames()

    def _load_frames(self):
        """
        Load video frames from the specified directory.

        Returns:
            list: A list of video frames.
        """
        frames = []
        if self.single_file:
            video_path = os.path.join(self.video_dir, self.single_file)
            frames.extend(self._extract_frames(video_path))
        else:
            for video_file in os.listdir(self.video_dir):
                if video_file.endswith((".mp4", ".avi")):
                    video_path = os.path.join(self.video_dir, video_file)
                    frames.extend(self._extract_frames(video_path))
        return frames

    def _extract_frames(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def __len__(self):
        """Return the total number of video frames."""
        return len(self.frames)

    def __getitem__(self, idx):
        """
        Get the video frame at the specified index.

        Args:
            idx (int): The index of the video frame.

        Returns:
            tuple: A tuple containing the video frame and its target.
        """
        frame = self.frames[idx]
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.transform:
            frame = self.transform(frame)

        # For fine-tuning, we'll use the original frame as both input and target
        return frame, frame

    @staticmethod
    def calculate_mean_std(dataset):
        mean = 0.0
        std = 0.0
        for frame, _ in dataset:
            mean += frame.mean(dim=[1, 2])
            std += frame.std(dim=[1, 2])
        mean /= len(dataset)
        std /= len(dataset)
        return mean, std

    @classmethod
    def get_normalization_transform(cls, dataset):
        mean, std = cls.calculate_mean_std(dataset)
        return transforms.Normalize(mean=mean.tolist(), std=std.tolist())
