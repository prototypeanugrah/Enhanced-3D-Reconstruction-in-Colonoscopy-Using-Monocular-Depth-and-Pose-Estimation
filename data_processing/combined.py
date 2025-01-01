"Module for the custom combined dataset"

from torch.utils import data
import torch
import numpy as np
import lightning as pl

from data_processing import simcol
from data_processing import c3vd


class CombinedDataset(data.Dataset):
    """
    Dataset class for the combined dataset.

    Args:
        simcol_dataset (SimColDataset): SimCol dataset instance
        c3vd_dataset (C3VDDataset): C3VD dataset instance
    """

    def __init__(
        self,
        simcol_dataset: simcol.SimColDataset,
        c3vd_dataset: c3vd.C3VDDataset,
    ) -> None:
        # Initialize individual datasets
        self.datasets = []

        if simcol_dataset is not None:
            self.datasets.append(simcol_dataset)

        if c3vd_dataset is not None:
            self.datasets.append(c3vd_dataset)

        if not self.datasets:
            raise ValueError("No datasets were provided to CombinedDataset")

        # Calculate lengths and cumulative lengths
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_lengths = np.cumsum(self.lengths)

        print(f"Mode: {c3vd_dataset.mode}")
        print(f"SimCol dataset length: {self.lengths[0]}")
        print(f"C3VD dataset length: {self.lengths[1]}")
        print(f"Total dataset length: {sum(self.lengths)}")

    def __len__(self) -> int:
        return sum(self.lengths)

    def __getitem__(
        self,
        idx: int,
    ) -> dict:
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
        result["source"] = torch.tensor(
            0.0 if dataset_idx == 0 else 1.0
        )  # 0 for SimCol, 1 for C3VD
        return result


class CombinedDataModule(pl.LightningDataModule):
    """
    Data module class for the combined dataset.

    Args:
        simcol_data_dir (str): Path to the SimCol dataset directory
        simcol_train_list (str): Path to the SimCol training list
        simcol_val_list (str): Path to the SimCol validation list
        simcol_test_list (str): Path to the SimCol test list
        c3vd_data_dir (str): Path to the C3VD dataset directory
        c3vd_train_list (str): Path to the C3VD training list
        c3vd_val_list (str): Path to the C3VD validation list
        c3vd_test_list (str): Path to the C3VD test list
        ds_type (str): Type of dataset (e.g. "rgb", "flow")
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        size (int): Size of the input images
    """

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
        ds_type: str,
        batch_size: int,
        num_workers: int,
        size: int,
    ) -> None:
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
        self.ds_type = ds_type

        # Initialize dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(
        self,
        stage: str,
    ) -> None:

        if stage == "fit" or stage is None:
            self.train_dataset = CombinedDataset(
                simcol_dataset=simcol.SimColDataset(
                    data_dir=self.simcol_data_dir,
                    data_list=self.simcol_train_list,
                    size=self.size,
                    hflip=True,
                    vflip=True,
                    mode="Train",
                    ds_type=self.ds_type,
                ),
                c3vd_dataset=c3vd.C3VDDataset(
                    data_dir=self.c3vd_data_dir,
                    data_list=self.c3vd_list,
                    size=self.size,
                    hflip=True,
                    vflip=True,
                    mode="Train",
                    ds_type=self.ds_type,
                ),
            )

            self.val_dataset = CombinedDataset(
                simcol_dataset=simcol.SimColDataset(
                    data_dir=self.simcol_data_dir,
                    data_list=self.simcol_val_list,
                    size=self.size,
                    mode="Val",
                    ds_type=self.ds_type,
                ),
                c3vd_dataset=c3vd.C3VDDataset(
                    data_dir=self.c3vd_data_dir,
                    data_list=self.c3vd_list,
                    size=self.size,
                    hflip=True,
                    vflip=True,
                    mode="Val",
                    ds_type=self.ds_type,
                ),
            )

        # if stage == "test" or stage is None:
        #     self.test_dataset = CombinedDataset(
        #         simcol_dataset=simcol.SimColDataset(
        #             data_dir=self.simcol_data_dir,
        #             data_list=self.simcol_test_list,
        #             size=self.size,
        #             mode="Test",
        #             ds_type=self.ds_type,
        #         )
        #         c3vd_dataset=c3vd.C3VDDataset(
        #             data_dir=self.c3vd_data_dir,
        #             data_list=self.c3vd_list,
        #             size=self.size,
        #             hflip=True,
        #             vflip=True,
        #             mode="Test",
        #             ds_type=self.ds_type,
        #         ),
        #     )

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
