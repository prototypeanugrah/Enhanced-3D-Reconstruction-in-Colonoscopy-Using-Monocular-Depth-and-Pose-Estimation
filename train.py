"Module for training/fine-tuning the DepthAnythingV2 model"

import argparse
import logging
import os

from torch.utils.tensorboard import SummaryWriter
import torch

from utils import utils
from training import training_utils
from data_processing import dataloader

# Set float32 matrix multiplication precision to 'high' for better performance
# on Tensor Cores
torch.set_float32_matmul_precision("high")

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
os.environ["HF_HOME"] = "~/home/public/avaishna/.cache"


def str2bool(v: str) -> bool:
    """
    Convert a string to a boolean.

    Args:
        v (str): The string to convert.

    Raises:
        argparse.ArgumentTypeError: If the string cannot be converted to a
        boolean.

    Returns:
        bool: The boolean value.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def count_parameters(model: torch.nn.Module) -> tuple:
    """
    Count the total and trainable parameters in a model.

    Args:
        model (torch.nn.Module): The model to count parameters for.

    Returns:
        tuple: A tuple containing the total parameters and trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        total_params,
        trainable_params,
    )


def main(
    input_path: str,
    output_path: str,
    batch_size: int,
    lr: float,
    model_size: str,
    epochs: int,
    use_scheduler: bool = True,
    warmup_steps: int = 500,
    logdir: str = None,
):

    val_vids = [
        os.path.join(input_path, "SyntheticColon_I/Frames_S4"),
        os.path.join(input_path, "SyntheticColon_I/Frames_S9"),
        os.path.join(input_path, "SyntheticColon_I/Frames_S14"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B4"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B9"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B14"),
    ]
    test_vids = [
        os.path.join(input_path, "SyntheticColon_I/Frames_S5"),
        os.path.join(input_path, "SyntheticColon_I/Frames_S10"),
        os.path.join(input_path, "SyntheticColon_I/Frames_S15"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B5"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B10"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B15"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O1"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O2"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O3"),
    ]

    # Get all subdirectories
    all_dirs = []
    for root, dirs, _ in os.walk(input_path):
        for dir in dirs:
            if dir.startswith("Frames_"):
                all_dirs.append(os.path.join(root, dir))

    # Create train_vids by excluding val_vids and test_vids
    train_vids = [
        dir for dir in all_dirs if dir not in val_vids and dir not in test_vids
    ]

    # Process the images and get the train and validation dataloaders
    train_depth, train_rgb, val_depth, val_rgb = utils.process_images(
        train_vids,
        val_vids,
        test_vids,
        input_path,
    )

    train_dataloader, val_dataloader = dataloader.get_dataloaders(
        train_depth,
        train_rgb,
        val_depth,
        val_rgb,
        batch_size=batch_size,
    )

    # Set up model
    model = training_utils.DepthEstimationModule(
        f"depth-anything/Depth-Anything-V2-{model_size}-hf",
        lr=lr,
        use_scheduler=use_scheduler,
        warmup_steps=warmup_steps,
    )

    # Count and print parameters
    total_params, trainable_params = count_parameters(model.model)
    logger.info("Total parameters: %d", total_params)
    logger.info("Trainable parameters: %d", trainable_params)

    custom_model_name = f"pretrained_l{lr}_e{epochs}_b{batch_size}_m{model_size}_s{1 if use_scheduler else 0}_w{warmup_steps if warmup_steps else 0}"

    # Set up TensorBoard writer
    writer = SummaryWriter(
        log_dir=os.path.join(
            logdir,
            custom_model_name,
        ),
    )

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(
        [p for n, p in model.named_parameters() if "lora_" in n],
        lr=lr,
    )
    scheduler = (
        training_utils.WarmupReduceLROnPlateau(
            optimizer,
            warmup_steps=warmup_steps,
            factor=0.5,
            patience=5,
        )
        if use_scheduler
        else None
    )

    # Initialize EarlyStopping
    early_stopping = training_utils.EarlyStopping(
        patience=5,
        verbose=True,
        path=os.path.join(output_path, f"{custom_model_name}_checkpoint.pt"),
    )

    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        train_loss = training_utils.train(
            model,
            train_dataloader,
            epoch,
            optimizer,
            device,
            writer,
        )

        val_loss = training_utils.validate(
            model,
            val_dataloader,
            epoch,
            device,
            writer,
        )

        logger.info(
            "Epoch %d/%d, Train Loss: %.4f",
            epoch + 1,
            epochs,
            train_loss,
        )
        logger.info(
            "Epoch %d/%d, Val Loss: %.4f",
            epoch + 1,
            epochs,
            val_loss,
        )

        # If using a scheduler, step it
        if scheduler:
            if isinstance(
                scheduler,
                training_utils.WarmupReduceLROnPlateau,
            ):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Current Learning Rate: {current_lr:.6f}")
        writer.add_scalar("Train/LearningRate", current_lr, epoch)

        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

    # Save the fine-tuned model
    custom_output_path = os.path.join(output_path, custom_model_name)
    model.model.save_pretrained(custom_output_path)
    logger.info("Saved fine-tuned model to %s", custom_output_path)

    # Close the TensorBoard writer
    writer.close()


# def main(
#     input_path: str,
#     output_path: str,
#     batch_size: int,
#     lr: float,
#     model_size: str,
#     epochs: int,
#     use_scheduler: bool = False,
#     warmup_steps: int = 500,
# ):
#     # Process the images and get the train and validation dataloaders
#     train_depth, train_rgb, val_depth, val_rgb = utils.process_images(
#         input_path,
#     )

#     train_dataloader, val_dataloader = utils.get_dataloaders(
#         train_depth,
#         train_rgb,
#         val_depth,
#         val_rgb,
#         batch_size=batch_size,
#     )

#     # temp_dataset = CustomFrameDepthDataset(
#     #     os.path.join(input_path, "val"),
#     #     transform=ransform,
#     # )

#     # normalize_transform = CustomFrameDepthDataset.get_normalization_transform(
#     #     temp_dataset
#     # )

#     # transform = transforms.Compose(
#     #     [
#     #         transform,
#     #         normalize_transform,
#     #     ]
#     # )

#     # train_dataset = CustomFrameDepthDataset(
#     #     os.path.join(input_path, "train"),
#     #     transform=transform,
#     # )
#     # else:

#     # train_dataset = CustomFrameDepthDataset(
#     #     # os.path.join(input_path, "train"),
#     #     input_path,
#     #     transform=transform,
#     #     start_idx=0,
#     #     end_idx=500,
#     #     # max_frames=1358,
#     # )
#     # train_dataset = Dataset(
#     #     input_paths=train_imgs,
#     #     target_paths=train_maps,
#     #     transform_input=transform_input,
#     #     transform_target=transform_target,
#     #     hflip=True,
#     #     vflip=True,
#     #     affine=False,
#     # )

#     # val_dataset = Dataset(
#     #     input_paths=val_imgs,
#     #     target_paths=val_maps,
#     #     transform_input=transform_input,
#     #     transform_target=transform_target,
#     # )

#     # val_dataset = CustomFrameDepthDataset(
#     #     # os.path.join(input_path, "val"),
#     #     input_path,
#     #     transform=transform,
#     #     start_idx=0,
#     #     end_idx=500,
#     #     # max_frames=1358,
#     # )
#     # train_dataloader = data.DataLoader(
#     #     train_dataset,
#     #     batch_size=batch_size,
#     #     drop_last=True,
#     #     shuffle=True,
#     #     num_workers=4,
#     # )
#     # logger.info("Train dataloader created and processed")

#     # val_dataloader = data.DataLoader(
#     #     val_dataset,
#     #     batch_size=batch_size,
#     #     drop_last=False,
#     #     shuffle=False,
#     #     num_workers=4,
#     # )
#     # logger.info("Validation dataloader created and processed")

#     # Set up model
#     model = DepthEstimationModule(
#         f"depth-anything/Depth-Anything-V2-{model_size}-hf",
#         lr=lr,
#         use_scheduler=use_scheduler,
#         warmup_steps=warmup_steps,
#     )

#     custom_model_name = f"pretrained_l{lr}_e{epochs}_b{batch_size}_m{model_size}_s{1 if use_scheduler else 0}_w{warmup_steps if warmup_steps else 0}"

#     # Set up TensorBoard logger
#     tb_logger = TensorBoardLogger(
#         "tb_logs",
#         name="depth_estimation",
#         version=custom_model_name,
#     )

#     # Set up trainer
#     trainer = pl.Trainer(
#         max_epochs=epochs,
#         accelerator="gpu" if torch.cuda.is_available() else "cpu",
#         devices=1,
#         logger=tb_logger,
#         callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
#         accumulate_grad_batches=2,
#         log_every_n_steps=1,
#     )

#     # Train the model
#     trainer.fit(
#         model,
#         train_dataloaders=train_dataloader,
#         val_dataloaders=val_dataloader,
#     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DepthAnythingV2 model on custom video dataset",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="Path to the directory containing input videos",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the fine-tuned model",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training",
    )
    parser.add_argument(
        "-m",
        "--model_size",
        type=str,
        default="small",
        choices=["Small", "Base", "Large"],
        help="Size of the DepthAnythingV2 model to use",
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "-s",
        "--use_scheduler",
        type=str,
        default="True",
        help="Learning Rate Scheduler",
    )
    parser.add_argument(
        "-w",
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "-ld",
        "--logdir",
        type=str,
        required=True,
        help="Tensorboard logging directory",
    )

    args = parser.parse_args()

    # Convert use_scheduler to boolean
    use_scheduler = str2bool(args.use_scheduler)

    os.makedirs(args.output_dir, exist_ok=True)  # Ensure the output directory exists
    os.makedirs(args.logdir, exist_ok=True)  # Ensure the logging directory exists

    main(
        args.input_dir,
        args.output_dir,
        args.batch_size,
        args.learning_rate,
        args.model_size,
        args.epochs,
        use_scheduler,
        args.warmup_steps,
        args.logdir,
    )
