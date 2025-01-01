"Module for the custom dataset"

import os
import glob
import random

from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as F
import cv2
import numpy as np
import lightning as pl
import torch


class C3VDDataset(data.Dataset):
    """
    Dataset class for the C3VD dataset

    Args:
        data_dir (str): Path to the dataset directory.
        data_list (str): Path to the list of data.
        size (int): Size of the image.
        hflip (bool): Horizontal flip.
        vflip (bool): Vertical flip.
        mode (str): Mode of the dataset. Can be 'Train', 'Val', or 'Test'.
        ds_type (str): Type of the dataset.
    """

    def __init__(
        self,
        data_dir: str,
        data_list: str,
        size: int,
        hflip: bool,
        vflip: bool,
        mode: str,
        ds_type: str,
    ) -> None:
        self.data_dir = data_dir
        self.size = size
        self.hflip = hflip
        self.vflip = vflip
        self.mode = mode
        self.ds_type = ds_type

        if self.mode in (
            "Train",
            "Val",
            # "Test",
        ):
            with open(data_list, "r") as f:
                content = f.read().strip()
                folders = [folder.strip() for folder in content.split(",")]

            self.images = []
            self.depths = []

            for folder in folders:
                if not folder:  # Skip empty strings
                    continue
                folder_path = os.path.join(self.data_dir, folder)

                if not os.path.exists(folder_path):
                    print(f"Warning: Folder does not exist: {folder_path}")
                    continue

                # Get all color images (both patterns)
                color_images = sorted(
                    glob.glob(
                        os.path.join(
                            folder_path,
                            "*_color.png",
                        )
                    )
                )
                color_images.extend(
                    sorted(
                        glob.glob(
                            os.path.join(
                                folder_path,
                                "[0-9]*_*.png",
                            )
                        )
                    )
                )

                # Get corresponding depth images
                valid_pairs = []
                valid_depths = []
                for img_path in color_images:
                    # Extract the base number from the filename
                    base_num = os.path.basename(img_path).split("_")[0]

                    # Try different possible depth filename patterns
                    depth_patterns = [
                        f"{base_num}_depth.tiff",  # original number
                        f"{int(base_num):04d}_depth.tiff",  # zero-padded to 4 digits
                    ]

                    depth_file = None
                    for pattern in depth_patterns:
                        candidate_path = os.path.join(
                            os.path.dirname(img_path), pattern
                        )
                        if os.path.exists(candidate_path):
                            depth_file = candidate_path
                            break

                    if depth_file:
                        valid_pairs.append(img_path)
                        valid_depths.append(depth_file)
                    else:
                        print(f"Warning: Missing depth file for {img_path}")

                self.images.extend(valid_pairs)
                self.depths.extend(valid_depths)

            assert len(self.images) == len(
                self.depths
            ), f"Mismatch in number of images and depths for {mode} set"

        else:
            raise ValueError("Mode must be one of: 'Train', 'Val', 'Test'")

        # if self.mode == "Train":
        self.transform_input = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.size, self.size),
                    antialias=True,
                    interpolation=cv2.INTER_CUBIC,
                ),
                # transforms.ColorJitter(
                #     brightness=0.4,  # Random brightness adjustment factor
                #     contrast=0.4,  # Random contrast adjustment factor
                #     saturation=0.2,  # Random saturation adjustment factor
                #     hue=0.1,  # Random hue adjustment factor
                # ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # else:
        #     # Keep the original transform for validation and test
        #     self.transform_input = transforms.Compose(
        #         [
        #             transforms.ToTensor(),
        #             transforms.Resize(
        #                 (self.size, self.size),
        #                 antialias=True,
        #             ),
        #             transforms.Normalize(
        #                 mean=[0.485, 0.456, 0.406],
        #                 std=[0.229, 0.224, 0.225],
        #             ),
        #         ]
        #     )

        self.transform_output = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.size, self.size),
                    antialias=True,
                ),
                # transforms.Lambda(lambda x: x / 100.0),
                # transforms.Normalize(
                #     mean=[0.280],
                #     std=[0.208],
                # ),  # Single channel normalization
                # Depth Mean (scaled): 0.2802938222885132
                # Depth Std (scaled): 0.20831811428070068
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> dict:
        info = self.images[idx].split(os.path.sep)
        dataset, frame_id = info[-3], info[-1].split(".")[0]

        image = cv2.imread(self.images[idx], cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(self.depths[idx], cv2.IMREAD_UNCHANGED)

        # Read image and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.dtype == np.uint16:
            # Convert np.uint16 to np.uint8
            image = (image / 256).astype("uint8")
        image = image.astype(np.float32) / 255.0

        # Read depth and scale appropriately
        depth = depth.astype(np.float32) / 65535.0  #  * 100.0

        # print(f"Depth range: {depth.min()} - {depth.max()}")
        # print(f"Mask range: {mask.min()} - {mask.max()}")

        image = self.transform_input(image)
        depth = self.transform_output(depth)

        # mask = (depth > 0.0) & (depth <= 100.0)  # .astype(np.float32)
        # mask = mask.to(torch.float32)

        mask = torch.isnan(depth) == 0
        depth[mask == 0] = 0

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
            "id": frame_id,
            "image": image,
            "depth": depth,
            "mask": mask,
            "ds_type": self.ds_type,
        }


class C3VDDataModule(pl.LightningDataModule):
    """
    Data module for the C3VD dataset

    Args:
        data_dir (str): Path to the dataset directory.
        train_list (str): Path to the training list.
        val_list (str): Path to the validation list.
        test_list (str): Path to the test list.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.
        size (int): Size of the image.
    """

    def __init__(
        self,
        data_dir: str,
        train_list: str,
        val_list: str,
        # test_list: str,
        ds_type: str,
        batch_size: int = 32,
        num_workers: int = 8,
        size: int = 518,
    ) -> None:
        super(C3VDDataModule, self).__init__()
        self.data_dir = data_dir
        self.train_list = train_list
        self.val_list = val_list
        # self.test_list = test_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size
        self.ds_type = ds_type

        # Initialize dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        # self.test_dataset = None

    def prepare_data(self) -> None:
        # This method is called only once and on 1 GPU
        pass

    def setup(
        self,
        stage: str | None = None,
    ) -> None:
        """
        Setup the dataset for the given stage.

        Args:
            stage (str | None, optional): Stage of the dataset. Can be 'fit',
            'test', or None. Defaults to None.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = C3VDDataset(
                data_dir=self.data_dir,
                data_list=self.train_list,
                mode="Train",
                size=self.size,
                hflip=True,
                vflip=True,
                ds_type=self.ds_type,
            )
            self.val_dataset = C3VDDataset(
                data_dir=self.data_dir,
                data_list=self.val_list,
                mode="Val",
                size=self.size,
                hflip=False,
                vflip=False,
                ds_type=self.ds_type,
            )
            if self.ds_type == "c3vd":
                print(f"C3VD Train: {len(self.train_dataset)}")
                print(f"C3VD Val:   {len(self.val_dataset)}")

        # if stage == "test" or stage is None:
        #     self.test_dataset = C3VDDataset(
        #         data_dir=self.data_dir,
        #         data_list=self.test_list,
        #         mode="Test",
        #         size=self.size,
        #         ds_type=self.ds_type,
        #     )

        #     if self.ds_type == "c3vd":
        #         print(f"C3VD Test:  {len(self.test_dataset)}\n")

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

    # def test_dataloader(self):
    #     return data.DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         num_workers=self.num_workers,
    #         pin_memory=True,
    #         persistent_workers=True,
    #         drop_last=False,
    #     )
