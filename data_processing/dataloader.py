"""
Module for creating the dataloaders.
"""

from torchvision import transforms
from torch.utils import data

from data_processing import dataset


def get_dataloaders(
    train_maps: list,
    train_imgs: list,
    val_maps: list,
    val_imgs: list,
    test_maps: list,
    test_imgs: list,
    batch_size: int,
) -> tuple:
    """
    Get the training and validation dataloaders.

    Args:
        train_maps (depth maps): The training depth maps.
        train_imgs (images): The training images.
        val_maps (depth maps): The validation depth maps.
        val_imgs (images): The validation images.
        batch_size (int): The batch size.

    Returns:
        tuple: A tuple containing the training and validation dataloaders.
    """
    transform_input = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (475, 475),
                antialias=True,
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # transform_target = transforms.ToTensor()
    transform_target = transforms.Compose(
        [
            transforms.Resize((475, 475)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = dataset.Dataset(
        input_paths=train_imgs,
        target_paths=train_maps,
        transform_input=transform_input,
        transform_target=transform_target,
        hflip=True,
        vflip=True,
        affine=False,
    )

    val_dataset = dataset.Dataset(
        input_paths=val_imgs,
        target_paths=val_maps,
        transform_input=transform_input,
        transform_target=transform_target,
    )

    test_dataset = dataset.Dataset(
        input_paths=test_imgs,
        target_paths=test_maps,
        transform_input=transform_input,
        transform_target=transform_target,
    )

    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True,  # Keep workers alive between epochs
    )

    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    test_dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
    )


def get_dataloaders_test(imgs):

    transform_input = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (475, 475),
                antialias=True,
            ),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_dataset = dataset.Dataset_test(
        input_paths=imgs,
        transform_input=transform_input,
    )

    dataloader = data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=32,
    )

    return dataloader
