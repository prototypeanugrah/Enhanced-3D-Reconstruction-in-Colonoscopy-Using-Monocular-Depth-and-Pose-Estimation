"Module for the Pose Estimation Pipeline"

import os

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
import numpy as np

from . import utils


class PoseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        data_list,
        mode,
        ds_type,
        size,
    ):
        self.data_dir = data_dir
        # self.positions = positions
        self.size = size

        self.mode = mode
        self.ds_type = ds_type

        # Read folder paths from text file
        if self.mode in ("Train", "Val", "Test"):
            with open(data_list, "r", encoding="utf-8") as f:
                folders = [folder.strip() for folder in f.read().strip().split(",")]

            # Get all frames from the folders
            self.rgb_paths = []
            self.depth_paths = []
            self.positions = []
            self.orientations = []

            for folder in folders:
                if not folder:  # Skip empty strings
                    continue
                folder_path = os.path.join(self.data_dir, folder)
                depth_frames, rgb_frames = utils.load_frames(folder_path)

                # Determine the position file prefix based on the dataset version
                dataset_version = folder.split("/")[
                    0
                ]  # Gets "SyntheticColon_I", "II", or "III"
                frame_number = folder.split("_")[-1]  # Gets the frame number

                # Set position file prefix based on dataset version
                if dataset_version == "SyntheticColon_III":
                    pos_prefix = "SavedPosition_O"
                elif dataset_version == "SyntheticColon_II":
                    pos_prefix = "SavedPosition_B"  # Adjust this based on actual prefix
                elif dataset_version == "SyntheticColon_I":
                    pos_prefix = "SavedPosition_S"  # Adjust this based on actual prefix
                else:
                    raise ValueError(f"Unknown dataset version: {dataset_version}")

                # Load positions from the corresponding SavedPosition file
                position_file = os.path.join(
                    self.data_dir,
                    dataset_version,
                    f"SavedPosition_{frame_number}.txt",
                )

                # Load orientations from the corresponding SavedOrientation file
                orientation_file = os.path.join(
                    self.data_dir,
                    dataset_version,
                    f"SavedRotationQuaternion_{frame_number}.txt",
                )

                try:
                    # Load positions and verify length matches number of frames
                    positions = np.loadtxt(
                        position_file
                    )  # Shape: (N, 3) for x,y,z positions

                    # Load orientations and verify length matches number of frames
                    orientations = np.loadtxt(
                        orientation_file
                    )  # Shape: (N, 4) for quaternion orientations

                    # Ensure positions is 2D array even if there's only one position
                    if positions.ndim == 1:
                        positions = positions.reshape(1, -1)

                    # Ensure orientations is 2D array even if there's only one orientation
                    if orientations.ndim == 1:
                        orientations = orientations.reshape(1, -1)

                    # Verify data shapes
                    if positions.shape[1] != 3:
                        raise ValueError(
                            f"Expected positions to have 3 columns, got {positions.shape[1]}"
                        )
                    if orientations.shape[1] != 4:
                        raise ValueError(
                            f"Expected quaternions to have 4 columns, got {orientations.shape[1]}"
                        )

                    # Verify number of positions matches number of frames
                    if len(positions) != len(rgb_frames) or len(orientations) != len(
                        rgb_frames
                    ):
                        print(
                            f"Warning: Mismatch in number of frames in {folder}:"
                            f" positions={len(positions)}, orientations={len(orientations)},"
                            f" rgb={len(rgb_frames)}"
                        )
                        # Use the minimum length to ensure alignment
                        min_len = min(len(positions), len(rgb_frames))
                        positions = positions[:min_len]
                        orientations = orientations[:min_len]
                        rgb_frames = rgb_frames[:min_len]
                        depth_frames = depth_frames[:min_len]

                    # Make a single positions list which is [position, orientation]
                    poses = np.concatenate(
                        (positions, orientations),
                        axis=1,
                    )  # Shape: (N,7)

                    # Extend all lists with aligned data
                    self.positions.extend(poses)
                    self.rgb_paths.extend(rgb_frames)
                    self.depth_paths.extend(depth_frames)

                except FileNotFoundError:
                    print(f"Warning: Position file not found: {position_file}")
                    continue
                except ValueError as e:
                    print(f"Warning: Error loading positions from {position_file}: {e}")
                    continue

            # Remove bad frames if in validation set
            if self.mode == "Val":
                self.rgb_paths, self.depth_paths, self.positions = (
                    utils.remove_bad_frames(
                        root_path=self.data_dir,
                        rgb_list=self.rgb_paths,
                        depth_list=self.depth_paths,
                        positions_list=self.positions,
                    )
                )

            # Verify data alignment
            assert (
                len(self.rgb_paths) == len(self.depth_paths) == len(self.positions)
            ), (
                f"Mismatch in number of images ({len(self.rgb_paths)}), "
                f"depths ({len(self.depth_paths)}), and positions ({len(self.positions)}) "
                f"for {self.mode} set"
            )

            # print(f"Loaded {len(self.rgb_paths)} frames for {self.mode} set")

        else:
            raise ValueError("Mode must be one of: 'Train', 'Val', 'Test'")

        self.transform_input = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.size, self.size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                # transforms.CenterCrop(self.size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    # mean=[0.646, 0.557, 0.473],
                    # std=[0.055, 0.046, 0.029],
                ),
            ]
        )

        self.transform_output = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (self.size, self.size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                # transforms.Normalize(
                #     # mean=[0.28444117],
                #     # std=[0.2079102],
                # ),  # Single channel normalization
            ]
        )

    def __len__(self):
        return len(self.rgb_paths) - 1  # Need pairs of consecutive frames

    def load_and_preprocess_rgb(self, path):
        # Load RGB image and convert to tensor
        # Normalize to [-1, 1] or [0, 1]
        image = np.array(Image.open(path))[:, :, :3]
        image = image.astype(np.float32) / 255.0  # Convert to [0,1] range
        image = self.transform_input(image)
        return image

    def load_and_preprocess_depth(self, path):
        # Load depth map and convert to tensor
        # Normalize appropriately
        depth = (
            np.array(Image.open(path)).astype(np.float32) / 65535.0
        )  # Convert to [0,1] range
        depth = self.transform_output(depth)
        return depth

    def __getitem__(self, idx):
        input_id = self.rgb_paths[idx]
        info = input_id.split(os.path.sep)
        dataset, index = info[2] + "/" + info[3], info[4]
        # target_id = self.target_paths[idx]

        # Load consecutive frames
        rgb1 = self.load_and_preprocess_rgb(self.rgb_paths[idx])  # Shape: (3, H, W)
        rgb2 = self.load_and_preprocess_rgb(self.rgb_paths[idx + 1])  # Shape: (3, H, W)

        depth1 = self.load_and_preprocess_depth(
            self.depth_paths[idx]
        )  # Shape: (1, H, W)
        depth2 = self.load_and_preprocess_depth(
            self.depth_paths[idx + 1]
        )  # Shape: (1, H, W)

        # Concatenate all channels
        frame1 = torch.cat([rgb1, depth1], dim=0)  # shape: [4, H, W]
        frame2 = torch.cat([rgb2, depth2], dim=0)  # shape: [4, H, W]
        # Final input: concatenate both frames along channels → 8 channels
        input_pair = torch.cat([frame1, frame2], dim=0)  # shape: [8, H, W]

        # Get positions and orientations
        pos1 = torch.tensor(
            self.positions[idx][:3], dtype=torch.float32
        )  # Just position
        pos2 = torch.tensor(self.positions[idx + 1][:3], dtype=torch.float32)

        # Get quaternions (already normalized)
        quat1 = torch.tensor(self.positions[idx][3:], dtype=torch.float32)
        quat2 = torch.tensor(self.positions[idx + 1][3:], dtype=torch.float32)

        # Calculate relative position and normalize by scale factor
        relative_pos = pos2 - pos1
        scale_factor = torch.norm(relative_pos) + 1e-8
        relative_pos = relative_pos / scale_factor

        # Calculate relative rotation (quaternion multiplication)
        # q_rel = q2 * q1^(-1)
        q1_inv = quat1 * torch.tensor(
            [-1, -1, -1, 1], dtype=torch.float32
        )  # Conjugate of unit quaternion

        # Calculate relative quaternion (q2 * q1_inv)
        # If q1 = [x1, y1, z1, w1] and q2 = [x2, y2, z2, w2], then:
        relative_quat = torch.zeros(4, dtype=torch.float32)
        # x = x2*w1 + y2*z1 - z2*y1 + w2*x1
        relative_quat[0] = (
            quat2[0] * q1_inv[3]
            + quat2[1] * q1_inv[2]
            - quat2[2] * q1_inv[1]
            + quat2[3] * q1_inv[0]
        )
        # y = -x2*z1 + y2*w1 + z2*x1 + w2*y1
        relative_quat[1] = (
            -quat2[0] * q1_inv[2]
            + quat2[1] * q1_inv[3]
            + quat2[2] * q1_inv[0]
            + quat2[3] * q1_inv[1]
        )
        # z = x2*y1 - y2*x1 + z2*w1 + w2*z1
        relative_quat[2] = (
            quat2[0] * q1_inv[1]
            - quat2[1] * q1_inv[0]
            + quat2[2] * q1_inv[3]
            + quat2[3] * q1_inv[2]
        )
        # w = -x2*x1 - y2*y1 - z2*z1 + w2*w1
        relative_quat[3] = (
            -quat2[0] * q1_inv[0]
            - quat2[1] * q1_inv[1]
            - quat2[2] * q1_inv[2]
            + quat2[3] * q1_inv[3]
        )

        # Ensure quaternion normalization is numerically stable
        relative_quat = F.normalize(relative_quat, dim=0, eps=1e-8)

        # Combine relative position and rotation
        relative_pose = torch.cat([relative_pos, relative_quat])

        sample = {
            "dataset": dataset,
            "id": index,
            "input": input_pair,  # shape: [8, H, W]
            "target": relative_pose,  # shape: [7]
        }
        return sample


class PoseDataModule(pl.LightningDataModule):
    """
    Data module class for the SimCol dataset.

    Args:
        data_dir (str): Path to the data directory.
        train_list (str): Path to the training list.
        val_list (str): Path to the validation list.
        test_list (str): Path to the test list.
        ds_type (str): Type of the dataset - "simcol".
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.
        size (int): Size of the input images.
    """

    def __init__(
        self,
        data_dir: str,
        train_list: str,
        val_list: str,
        test_list: str,
        ds_type: str,
        batch_size: int,
        num_workers: int,
        size: int,
    ):

        super(PoseDataModule, self).__init__()
        self.data_dir = data_dir
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
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
            self.train_dataset = PoseDataset(
                data_dir=self.data_dir,
                data_list=self.train_list,
                size=self.size,
                mode="Train",
                ds_type=self.ds_type,
            )
            self.val_dataset = PoseDataset(
                data_dir=self.data_dir,
                data_list=self.val_list,
                size=self.size,
                mode="Val",
                ds_type=self.ds_type,
            )

            if self.ds_type == "simcol":
                print(f"SimCol Train: {len(self.train_dataset)}")
                print(f"SimCol Val:   {len(self.val_dataset)}")

        if stage == "test" or stage is None:
            self.test_dataset = PoseDataset(
                data_dir=self.data_dir,
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
