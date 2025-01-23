import torch
from tqdm import tqdm
from torchvision import transforms

from data_processing.simcol import SimColDataModule
from data_processing.c3vd import C3VDDataModule


def get_transform_without_normalize(size):
    """Create transform pipeline without normalization."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (size, size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
        ]
    )


def calculate_stats(dataloader, is_depth=False):
    """Calculate mean and std of the dataset."""
    channels = 1 if is_depth else 3
    psum = torch.zeros(channels)
    psum_sq = torch.zeros(channels)
    pixel_count = 0

    # First pass: mean
    for batch in tqdm(dataloader, desc="Calculating mean"):
        if is_depth:
            data = batch["depth"]
            # Only consider valid depth values (non-zero and non-nan)
            mask = (batch["mask"] > 0) & (~torch.isnan(data))
            data = data[mask]
        else:
            data = batch["image"]
            # Denormalize if using ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(data.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(data.device)
            data = data * std + mean

        batch_samples = (
            data.size(0) if is_depth else data.size(0) * data.size(2) * data.size(3)
        )
        pixel_count += batch_samples

        if is_depth:
            psum[0] += data.sum()
            psum_sq[0] += (data**2).sum()
        else:
            psum += data.sum(dim=[0, 2, 3])
            psum_sq += (data**2).sum(dim=[0, 2, 3])

    mean = psum / pixel_count
    var = (psum_sq / pixel_count) - (mean**2)
    std = torch.sqrt(var)

    return mean.numpy(), std.numpy()


def main():
    # SimCol Dataset
    simcol_dm = SimColDataModule(
        data_dir="./datasets/SyntheticColon/",
        train_list="./datasets/SyntheticColon/train.txt",
        val_list="./datasets/SyntheticColon/val.txt",
        test_list="./datasets/SyntheticColon/test.txt",
        ds_type="simcol",
        batch_size=32,
        num_workers=4,
        size=518,
    )
    simcol_dm.setup("fit")

    # Remove normalization from transforms temporarily
    simcol_dm.train_dataset.transform_input = get_transform_without_normalize(
        simcol_dm.size
    )
    simcol_dm.train_dataset.transform_target = get_transform_without_normalize(
        simcol_dm.size
    )

    print("\nCalculating SimCol RGB stats...")
    rgb_mean, rgb_std = calculate_stats(simcol_dm.train_dataloader())
    print("\nCalculating SimCol Depth stats...")
    depth_mean, depth_std = calculate_stats(
        simcol_dm.train_dataloader(),
        is_depth=True,
    )

    print("\nSimCol Dataset Statistics:")
    print(f"RGB Mean: {rgb_mean}")
    print(f"RGB Std:  {rgb_std}")
    print(f"Depth Mean: {depth_mean}")
    print(f"Depth Std:  {depth_std}")

    # C3VD Dataset
    c3vd_dm = C3VDDataModule(
        data_dir="./datasets/C3VD/",
        train_list="./datasets/C3VD/train.txt",
        val_list="./datasets/C3VD/val.txt",
        ds_type="c3vd",
        batch_size=32,
        num_workers=4,
        size=518,
    )
    c3vd_dm.setup("fit")

    # Remove normalization from transforms temporarily
    c3vd_dm.train_dataset.transform_input = get_transform_without_normalize(
        c3vd_dm.size
    )
    c3vd_dm.train_dataset.transform_output = get_transform_without_normalize(
        c3vd_dm.size
    )

    print("\nCalculating C3VD RGB stats...")
    rgb_mean, rgb_std = calculate_stats(c3vd_dm.train_dataloader())
    print("\nCalculating C3VD Depth stats...")
    depth_mean, depth_std = calculate_stats(
        c3vd_dm.train_dataloader(),
        is_depth=True,
    )

    print("\nC3VD Dataset Statistics:")
    print(f"RGB Mean: {rgb_mean}")
    print(f"RGB Std:  {rgb_std}")
    print(f"Depth Mean: {depth_mean}")
    print(f"Depth Std:  {depth_std}")


if __name__ == "__main__":
    main()
