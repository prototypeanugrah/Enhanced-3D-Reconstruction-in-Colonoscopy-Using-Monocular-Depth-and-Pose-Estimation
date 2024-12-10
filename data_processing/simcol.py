"Module for the custom dataset"

import os
import random

from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as F
import cv2
import numpy as np
import lightning as pl

from utils import utils


class SimColDataset(data.Dataset):
    """
    Dataset class for the custom dataset.

    Args:
        data (list): List of input and target paths.
    """

    def __init__(
        self,
        data_dir: str,
        data_list: str,
        size: int = 518,
        hflip: bool = False,
        vflip: bool = False,
        mode: str = None,
        ds_type: str = None,
    ):

        self.data_dir = data_dir
        self.size = size
        self.hflip = hflip
        self.vflip = vflip
        self.ds_type = ds_type

        # Read folder paths from text file
        if mode in ("Train", "Val", "Test"):
            with open(data_list, "r") as f:
                folders = [folder.strip() for folder in f.read().strip().split(",")]

            # Get all frames from the folders
            self.input_paths = []
            self.target_paths = []

            for folder in folders:
                if not folder:  # Skip empty strings
                    continue
                folder_path = os.path.join(self.data_dir, folder)
                depth_frames, rgb_frames = utils.load_frames(folder_path)

                self.input_paths.extend(rgb_frames)
                self.target_paths.extend(depth_frames)

            # Remove bad frames if in validation set
            if mode == "Val":
                self.input_paths, self.target_paths = utils.remove_bad_frames(
                    self.input_paths,
                    self.target_paths,
                    self.data_dir,
                )

            assert len(self.input_paths) == len(
                self.target_paths
            ), f"Mismatch in number of images and depths for {mode} set"

        else:
            raise ValueError("Mode must be one of: 'Train', 'Val', 'Test'")

        self.transform_input = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.size, self.size),
                    # interpolation=cv2.INTER_CUBIC,
                    antialias=True,
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.transform_target = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.size, self.size),
                    # interpolation=cv2.INTER_CUBIC,
                    antialias=True,
                ),
                # transforms.Normalize(
                #     mean=[0.5],
                #     std=[0.5],
                # ),
            ]
        )

    def __len__(self) -> int:
        return len(self.input_paths)

    def _download(self):
        raise NotImplementedError("Download not supported for this dataset")

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
        info = input_id.split(os.path.sep)
        dataset, idx = info[2] + "/" + info[3], info[4]
        target_id = self.target_paths[index]

        image = cv2.imread(input_id, cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(target_id, cv2.IMREAD_UNCHANGED).astype(np.float32)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.dtype == np.uint16:
            image = (image / 256).astype("uint8")
        image = image.astype(np.float32) / 255.0

        # Convert from 16-bit integer [0, 65535] to centimeters [0, 20.0]
        depth = (depth / 65536.0) * 20.0

        image = self.transform_input(image)
        depth = self.transform_target(depth)

        if self.hflip:
            if random.uniform(0.0, 1.0) > 0.5:
                image = F.hflip(image)
                depth = F.hflip(depth)

        if self.vflip:
            if random.uniform(0.0, 1.0) > 0.5:
                image = F.vflip(image)
                depth = F.vflip(depth)

        return {
            "dataset": dataset,
            "id": idx,
            "image": image,
            "depth": depth,
            "ds_type": self.ds_type,
        }

    # def plot(
    #     self,
    #     image,
    #     depth,
    #     prediction=None,
    #     show_titles=True,
    # ):
    #     if prediction is not None:
    #         prediction = prediction.clip(0, 1).float()
    #     # Convert image to [0, 1] range
    #     image = image.float()
    #     image = image - image.min()
    #     image = image / image.max()

    #     depth = depth.float()

    #     showing_prediction = prediction is not None
    #     ncols = 2 + int(showing_prediction)
    #     fig, axs = plt.subplots(
    #         nrows=1,
    #         ncols=ncols,
    #         figsize=(ncols * 4, 4),
    #     )
    #     axs[0].imshow(image.permute(1, 2, 0))
    #     axs[0].axis("off")
    #     axs[1].imshow(
    #         depth.squeeze(),
    #         cmap="Spectral_r",
    #     )
    #     axs[1].axis("off")
    #     if show_titles:
    #         axs[0].set_title("Image")
    #         axs[1].set_title("Depth")

    #     if showing_prediction:
    #         axs[2].imshow(
    #             prediction.squeeze(),
    #             cmap="Spectral_r",
    #         )
    #         axs[2].axis("off")
    #         if show_titles:
    #             axs[2].set_title("Prediction")
    #     return fig


class SimColDataModule(pl.LightningDataModule):

    def __init__(
        self,
        simcol_data_dir: str,
        simcol_train_list: str,
        simcol_val_list: str,
        simcol_test_list: str,
        ds_type: str = None,
        batch_size: int = 32,
        num_workers: int = 8,
        size: int = 518,
    ):

        super(SimColDataModule, self).__init__()
        self.simcol_data_dir = simcol_data_dir
        self.train_list = simcol_train_list
        self.val_list = simcol_val_list
        self.test_list = simcol_test_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.ds_type = ds_type

        # Initialize dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        # This method is called only once and on 1 GPU
        # Use this method to download data or perform one-time operations
        pass

    def setup(
        self,
        stage: str | None = None,
    ) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = SimColDataset(
                data_dir=self.simcol_data_dir,
                data_list=self.train_list,
                size=self.size,
                hflip=True,
                vflip=True,
                mode="Train",
                ds_type=self.ds_type,
            )
            self.val_dataset = SimColDataset(
                data_dir=self.simcol_data_dir,
                data_list=self.val_list,
                size=self.size,
                mode="Val",
                ds_type=self.ds_type,
            )

            if self.ds_type == "simcol":
                print(f"SimCol Train: {len(self.train_dataset)}")
                print(f"SimCol Val:   {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            self.test_dataset = SimColDataset(
                data_dir=self.simcol_data_dir,
                data_list=self.test_list,
                size=self.size,
                mode="Test",
                ds_type=self.ds_type,
            )

            if self.ds_type == "simcol":
                print(f"SimCol Test: {len(self.test_dataset)}")

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )
