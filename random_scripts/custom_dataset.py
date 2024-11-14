"Module for custom dataset"

import os

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

import cv2
import numpy as np
import torch


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
        max_frames=None,
    ):
        self.video_dir = video_dir
        self.transform = transform
        self.single_file = single_file
        self.max_frames = max_frames

        # Load video file paths, but do not load frames yet
        if self.single_file:
            self.video_files = [os.path.join(self.video_dir, self.single_file)]
        else:
            self.video_files = [
                os.path.join(self.video_dir, f)
                for f in os.listdir(self.video_dir)
                if f.endswith((".mp4", ".avi"))
            ]
        # self.frames = self._load_frames()

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
            video_files = [
                f for f in os.listdir(self.video_dir) if f.endswith((".mp4", ".avi"))
            ]
            for video_file in tqdm(video_files, desc="Loading videos", unit="video"):
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
        return len(self.video_files)

    def __getitem__(self, idx):
        """
        Get the video frame at the specified index.

        Args:
            idx (int): The index of the video frame.

        Returns:
            tuple: A tuple containing the video frame and its target.
        """
        video_path = self.video_files[idx]

        # Lazily load frames for the current video file
        frames = self._extract_frames(video_path)

        if self.max_frames is not None:
            if len(frames) > self.max_frames:
                # Truncate or randomly sample
                frames = frames[: self.max_frames]
                # frames = random.sample(frames, self.max_frames)
            elif len(frames) < self.max_frames:
                # Pad with zeros
                padding = [
                    np.zeros_like(frames[0])
                    for _ in range(self.max_frames - len(frames))
                ]
                frames.extend(padding)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack frames into a single tensor
        frames_tensor = torch.stack(
            [torch.from_numpy(f).float().to(torch.float16) for f in frames]
        )

        # For fine-tuning, we'll use the original frame as both input and target
        return frames_tensor, frames_tensor

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
