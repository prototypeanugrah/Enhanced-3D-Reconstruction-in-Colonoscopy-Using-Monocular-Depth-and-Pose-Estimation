"Module for the custom dataset"

import os
import glob

from torch.utils import data

import cv2

# import matplotlib.pyplot as plt
import numpy as np
import lightning as pl
import torch
from torchvision import transforms

from utils import utils


class SimColDataset(data.Dataset):
    """
    Dataset class for the custom dataset.

    Args:
        data (list): List of input and target paths.
    """

    def __init__(
        self,
        # input_paths: list,
        # target_paths: list,
        # size: int = 518,
        data_dir: str,
        data_list: str,
        size: int = 518,
        mode: str = None,
    ):
        # self.input_paths = input_paths
        # self.target_paths = target_paths
        # self.size = size

        self.data_dir = data_dir
        self.size = size

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
                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5],
                ),
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

        return {
            "dataset": dataset,
            "id": idx,
            "image": image,
            "depth": depth,
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
        # input_path: str,
        # batch_size: int = 32,
        # num_workers: int = 8,
        # size: int = 518,
        data_dir: str,
        train_list: str,
        val_list: str,
        test_list: str,
        batch_size: int = 32,
        num_workers: int = 8,
        size: int = 518,
    ):
        # super(SimColDataModule, self).__init__()
        # self.input_path = input_path
        # self.batch_size = batch_size
        # self.num_workers = num_workers
        # self.size = size

        super(SimColDataModule, self).__init__()
        self.data_dir = data_dir
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size

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
        # Define validation and test videos
        # val_vids = [
        #     os.path.join(self.input_path, "SyntheticColon_I/Frames_S4"),
        #     os.path.join(self.input_path, "SyntheticColon_I/Frames_S9"),
        #     os.path.join(self.input_path, "SyntheticColon_I/Frames_S14"),
        #     os.path.join(self.input_path, "SyntheticColon_II/Frames_B4"),
        #     os.path.join(self.input_path, "SyntheticColon_II/Frames_B9"),
        #     os.path.join(self.input_path, "SyntheticColon_II/Frames_B14"),
        # ]

        # test_vids = [
        #     os.path.join(self.input_path, "SyntheticColon_I/Frames_S5"),
        #     os.path.join(self.input_path, "SyntheticColon_I/Frames_S10"),
        #     os.path.join(self.input_path, "SyntheticColon_I/Frames_S15"),
        #     os.path.join(self.input_path, "SyntheticColon_II/Frames_B5"),
        #     os.path.join(self.input_path, "SyntheticColon_II/Frames_B10"),
        #     os.path.join(self.input_path, "SyntheticColon_II/Frames_B15"),
        #     os.path.join(self.input_path, "SyntheticColon_III/Frames_O1"),
        #     os.path.join(self.input_path, "SyntheticColon_III/Frames_O2"),
        #     os.path.join(self.input_path, "SyntheticColon_III/Frames_O3"),
        # ]

        # # Get all subdirectories
        # all_dirs = []
        # for root, dirs, _ in os.walk(self.input_path):
        #     for dir in dirs:
        #         if dir.startswith("Frames_"):
        #             all_dirs.append(os.path.join(root, dir))

        # # Create train_vids
        # train_vids = [
        #     dir for dir in all_dirs if dir not in val_vids and dir not in test_vids
        # ]

        # # Process images
        # train_depth, train_rgb, val_depth, val_rgb, test_depth, test_rgb = (
        #     utils.process_images(
        #         train_vids,
        #         val_vids,
        #         test_vids,
        #         self.input_path,
        #     )
        # )

        # Create datasets
        # if stage == "fit" or stage is None:
        #     self.train_dataset = SimColDataset(
        #         input_paths=train_rgb,
        #         target_paths=train_depth,
        #         size=self.size,
        #     )

        #     self.val_dataset = SimColDataset(
        #         input_paths=val_rgb,
        #         target_paths=val_depth,
        #         size=self.size,
        #     )

        # if stage == "test" or stage is None:
        #     self.test_dataset = SimColDataset(
        #         input_paths=test_rgb,
        #         target_paths=test_depth,
        #         size=self.size,
        #     )
        if stage == "fit" or stage is None:
            self.train_dataset = SimColDataset(
                data_dir=self.data_dir,
                data_list=self.train_list,
                size=self.size,
                mode="Train",
            )
            self.val_dataset = SimColDataset(
                data_dir=self.data_dir,
                data_list=self.val_list,
                size=self.size,
                mode="Val",
            )

        if stage == "test" or stage is None:
            self.test_dataset = SimColDataset(
                data_dir=self.data_dir,
                data_list=self.test_list,
                size=self.size,
                mode="Test",
            )

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
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class C3VD_Dataset(data.Dataset):
    """
    Dataset class for the C3VD dataset

    Args:
        data_dir (str): Path to the dataset directory.
        data_list (str): Path to the list of data.
        size (int): Size of the image.
        mode (str): Mode of the dataset. Can be 'Train', 'Val', or 'Test'.
    """

    def __init__(
        self,
        data_dir: str,
        data_list,
        size: int = 518,
        mode=None,
    ):
        self.data_dir = data_dir
        self.size = size

        # if mode in ("Train", "Val"):
        #     with open(data_list, "r") as f:
        #         self.dirs = f.readline().strip().split(" ")
        # elif mode == "Test":
        #     self.dirs = data_list
        # else:
        #     raise ValueError(
        #         "Mode not set or incorrect mode set! Only 'Train' or 'Test' is valid!"
        #     )

        # self.images = []
        # self.depths = []

        # for path in self.dirs:
        #     images = sorted(
        #         glob.glob(
        #             os.path.join(
        #                 self.data_dir,
        #                 path,
        #                 "images",
        #                 "*.png",
        #             )
        #         )
        #     )
        #     depths = sorted(
        #         glob.glob(
        #             os.path.join(
        #                 self.data_dir,
        #                 path,
        #                 "depths",
        #                 "*.tiff",
        #             )
        #         )
        #     )

        #     self.images.extend(images)
        #     self.depths.extend(depths)

        if mode in ("Train", "Val", "Test"):
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
                    glob.glob(os.path.join(folder_path, "*_color.png"))
                )
                color_images.extend(
                    sorted(glob.glob(os.path.join(folder_path, "[0-9]*_*.png")))
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

        self.transform_input = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.size, self.size),
                    antialias=True,
                ),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.transform_output = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.size, self.size),
                    antialias=True,
                ),
                transforms.Normalize(
                    mean=[0.5],
                    std=[0.5],
                ),  # Single channel normalization
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
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
        depth = (depth.astype(np.float32) / 65535.0) * 100.0

        # min_depth = 0.0
        # max_depth = 1.0

        # Create a mask to enable only depth values within the range
        # mask = ((depth >= min_depth) & (depth <= max_depth)).astype(np.float32)

        # Preprocessing
        # image = self.transform(image)
        # depth = self.transform(depth)
        # mask = self.transform(mask)

        image = self.transform_input(image)
        depth = self.transform_output(depth)

        return {
            "dataset": dataset,
            "id": frame_id,
            "image": image,
            "depth": depth,
            # "mask": mask,
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
        test_list: str,
        batch_size: int = 32,
        num_workers: int = 8,
        size: int = 518,
    ):
        super(C3VDDataModule, self).__init__()
        self.data_dir = data_dir
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size

        # Initialize dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

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
            self.train_dataset = C3VD_Dataset(
                data_dir=self.data_dir,
                data_list=self.train_list,
                mode="Train",
                size=self.size,
            )
            self.val_dataset = C3VD_Dataset(
                data_dir=self.data_dir,
                data_list=self.val_list,
                mode="Val",
                size=self.size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = C3VD_Dataset(
                data_dir=self.data_dir,
                data_list=self.test_list,
                mode="Test",
                size=self.size,
            )

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
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class CombinedDataset(data.Dataset):
    """
    Dataset class for the combined dataset.

    Args:
        simcol_dataset (SimColDataset): SimCol dataset instance
        c3vd_data_dir (str): Path to the C3VD dataset directory
        c3vd_list (str): Path to the list of data for C3VD dataset
        size (int): Size of the image
        mode (str): Mode of the dataset. Can be 'Train', 'Val', or 'Test'
    """

    def __init__(
        self,
        simcol_dataset: SimColDataset = None,
        c3vd_data_dir: str = None,
        c3vd_list: str = None,
        size: int = 518,
        mode: str = None,
    ):
        self.size = size
        self.mode = mode

        # Initialize individual datasets
        self.datasets = []

        if simcol_dataset is not None:
            self.datasets.append(simcol_dataset)
            # print(f"Added SimCol dataset with {len(simcol_dataset)} samples")

        if c3vd_data_dir is not None and c3vd_list is not None:
            c3vd_dataset = C3VD_Dataset(
                data_dir=c3vd_data_dir,
                data_list=c3vd_list,
                size=size,
                mode=mode,
            )
            self.datasets.append(c3vd_dataset)
            # print(f"Added C3VD dataset with {len(c3vd_dataset)} samples")

        if not self.datasets:
            raise ValueError("No datasets were provided to CombinedDataset")

        # Calculate lengths and cumulative lengths
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)

        print(f"Mode: {mode} ----- Individual dataset lengths: {self.lengths}")

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        # Find which dataset the index belongs to
        dataset_idx = np.searchsorted(
            self.cumulative_lengths,
            idx,
            side="right",
        )

        # Calculate the local index for the selected dataset
        local_idx = idx
        if dataset_idx > 0:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]

        # Convert to integer to ensure proper indexing
        dataset_idx = int(dataset_idx)
        local_idx = int(local_idx)

        # Get the item from the appropriate dataset using the local index
        result = self.datasets[dataset_idx][local_idx]
        result["source"] = torch.tensor(1.0 if dataset_idx == 0 else 0.0)
        return result


class CombinedDataModule(pl.LightningDataModule):

    def __init__(
        self,
        simcol_data_dir: str,
        simcol_train_list: str,
        simcol_val_list: str,
        simcol_test_list: str,
        c3vd_data_dir: str,
        c3vd_train_list: str,
        c3vd_val_list: str,
        c3vd_test_list: str,
        batch_size: int = 32,
        num_workers: int = 8,
        size: int = 518,
    ):
        super(CombinedDataModule, self).__init__()

        # SimCol parameters
        self.simcol_data_dir = simcol_data_dir
        self.simcol_train_list = simcol_train_list
        self.simcol_val_list = simcol_val_list
        self.simcol_test_list = simcol_test_list

        # C3VD parameters
        self.c3vd_data_dir = c3vd_data_dir
        self.c3vd_train_list = c3vd_train_list
        self.c3vd_val_list = c3vd_val_list
        self.c3vd_test_list = c3vd_test_list

        # Common parameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.size = size

        # Initialize dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        # Get SimCol splits
        # val_vids = [
        #     os.path.join(self.simcol_path, "SyntheticColon_I/Frames_S4"),
        #     os.path.join(self.simcol_path, "SyntheticColon_I/Frames_S9"),
        #     os.path.join(self.simcol_path, "SyntheticColon_I/Frames_S14"),
        #     os.path.join(self.simcol_path, "SyntheticColon_II/Frames_B4"),
        #     os.path.join(self.simcol_path, "SyntheticColon_II/Frames_B9"),
        #     os.path.join(self.simcol_path, "SyntheticColon_II/Frames_B14"),
        # ]

        # test_vids = [
        #     os.path.join(self.simcol_path, "SyntheticColon_I/Frames_S5"),
        #     os.path.join(self.simcol_path, "SyntheticColon_I/Frames_S10"),
        #     os.path.join(self.simcol_path, "SyntheticColon_I/Frames_S15"),
        #     os.path.join(self.simcol_path, "SyntheticColon_II/Frames_B5"),
        #     os.path.join(self.simcol_path, "SyntheticColon_II/Frames_B10"),
        #     os.path.join(self.simcol_path, "SyntheticColon_II/Frames_B15"),
        #     os.path.join(self.simcol_path, "SyntheticColon_III/Frames_O1"),
        #     os.path.join(self.simcol_path, "SyntheticColon_III/Frames_O2"),
        #     os.path.join(self.simcol_path, "SyntheticColon_III/Frames_O3"),
        # ]

        # # Get all subdirectories
        # all_dirs = []
        # for root, dirs, _ in os.walk(self.simcol_path):
        #     for dir in dirs:
        #         if dir.startswith("Frames_"):
        #             all_dirs.append(os.path.join(root, dir))

        # train_vids = [
        #     dir for dir in all_dirs if dir not in val_vids and dir not in test_vids
        # ]

        # # Process SimCol images
        # train_depth, train_rgb, val_depth, val_rgb, test_depth, test_rgb = (
        #     utils.process_images(
        #         train_vids,
        #         val_vids,
        #         test_vids,
        #         self.simcol_path,
        #     )
        # )

        if stage == "fit" or stage is None:
            self.train_dataset = CombinedDataset(
                simcol_dataset=SimColDataset(
                    data_dir=self.simcol_data_dir,
                    data_list=self.simcol_train_list,
                    size=self.size,
                    mode="Train",
                ),
                c3vd_data_dir=self.c3vd_data_dir,
                c3vd_list=self.c3vd_train_list,
                size=self.size,
                mode="Train",
            )

            self.val_dataset = CombinedDataset(
                simcol_dataset=SimColDataset(
                    data_dir=self.simcol_data_dir,
                    data_list=self.simcol_val_list,
                    size=self.size,
                    mode="Val",
                ),
                c3vd_data_dir=self.c3vd_data_dir,
                c3vd_list=self.c3vd_val_list,
                size=self.size,
                mode="Val",
            )

        if stage == "test" or stage is None:
            self.test_dataset = CombinedDataset(
                simcol_dataset=SimColDataset(
                    data_dir=self.simcol_data_dir,
                    data_list=self.simcol_test_list,
                    size=self.size,
                    mode="Test",
                ),
                c3vd_data_dir=self.c3vd_data_dir,
                c3vd_list=self.c3vd_test_list,
                size=self.size,
                mode="Test",
            )

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
        )

    def test_dataloader(self):
        return data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
