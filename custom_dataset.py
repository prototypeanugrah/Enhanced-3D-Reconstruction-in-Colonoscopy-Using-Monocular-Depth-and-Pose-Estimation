import os

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import torch


class CustomVideoDepthDataset(Dataset):
    def __init__(
        self, video_dir, transform=None
    ):
        self.video_dir = video_dir
        self.transform = transform
        self.frames = self._load_frames()

    def _load_frames(self):
        frames = []
        for video_file in os.listdir(self.video_dir):
            if video_file.endswith((".mp4", ".avi")):
                cap = cv2.VideoCapture(os.path.join(self.video_dir, video_file))
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.transform:
            frame = self.transform(frame)

        # For fine-tuning, we'll use the original frame as both input and target
        return frame, frame
