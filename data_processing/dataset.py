"Module for the custom dataset"

import random

from PIL import Image
from torch.utils import data

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF


class Dataset(data.Dataset):
    """
    Dataset class for the custom dataset.

    Args:
        data (list): List of input and target paths.
    """

    def __init__(
        self,
        input_paths: list,
        target_paths: list,
        transform_input: torchvision.transforms = None,
        transform_target: torchvision.transforms = None,
        hflip: bool = False,
        vflip: bool = False,
        affine: bool = False,
    ):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.hflip = hflip
        self.vflip = vflip
        self.affine = affine

    def __len__(self) -> int:
        return len(self.input_paths)

    def __getitem__(
        self,
        index: int,
    ) -> tuple:
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to get.

        Returns:
            tuple: A tuple containing the input image and target depth map.
        """
        input_id = self.input_paths[index]
        target_id = self.target_paths[index]

        # Change this line
        # x = Image.open(input_id).convert("RGB")
        x = cv2.imread(input_id)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        # y = np.array(Image.open(target_id)) / 255 / 256
        y = cv2.imread(target_id, cv2.IMREAD_UNCHANGED).astype(
            np.float32
        )  # Read as-is without any conversion
        y /= 255.0 * 256.0  # Convert to float and normalize

        # Resize y to 350x350 using OpenCV
        y = cv2.resize(
            y,
            (350, 350),
            interpolation=cv2.INTER_CUBIC,
        )

        # Apply transforms
        if self.transform_input:
            x = self.transform_input(x)

        # Convert y to tensor
        y = torch.from_numpy(y).float()

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.hflip(x)
                y = TF.hflip(y)

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                x = TF.vflip(x)
                y = TF.vflip(y)

        if self.affine:
            angle = random.uniform(-180.0, 180.0)
            h_trans = random.uniform(-350 / 8, 350 / 8)
            v_trans = random.uniform(-350 / 8, 350 / 8)
            scale = random.uniform(0.5, 1.5)
            shear = random.uniform(-22.5, 22.5)
            x = TF.affine(
                x,
                angle,
                (h_trans, v_trans),
                scale,
                shear,
                fill=-1.0,
            )
            y = TF.affine(
                y,
                angle,
                (h_trans, v_trans),
                scale,
                shear,
                fill=0.0,
            )
        return (
            x.float(),
            y.float(),
        )


class Dataset_test(data.Dataset):
    def __init__(self, input_paths: list, transform_input=None):
        self.input_paths = input_paths
        self.transform_input = transform_input

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index: int):
        input_ID = self.input_paths[index]

        x = np.array(Image.open(input_ID))[:, :, :3]
        x = self.transform_input(x)

        return x.float()
