"Module for training/fine-tuning the DepthAnythingV2 model"

import argparse
import logging
import os

from lightning_model import DepthEstimationModule
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import pytorch_lightning as pl

# import custom_dataset_frames
import utils

# Set float32 matrix multiplication precision to 'high' for better performance
# on Tensor Cores
torch.set_float32_matmul_precision("high")

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
os.environ["HF_HOME"] = "~/home/public/avaishna/.cache"


def main(
    input_path: str,
    output_path: str,
    batch_size: int,
    lr: float,
    model_size: str,
    epochs: int,
    use_scheduler: bool = False,
    warmup_steps: int = 500,
):
    # Process the images and get the train and validation dataloaders
    train_depth, train_rgb, val_depth, val_rgb = utils.process_images(
        input_path,
    )

    train_dataloader, val_dataloader = utils.get_dataloaders(
        train_depth,
        train_rgb,
        val_depth,
        val_rgb,
        batch_size=batch_size,
    )

    # temp_dataset = CustomFrameDepthDataset(
    #     os.path.join(input_path, "val"),
    #     transform=ransform,
    # )

    # normalize_transform = CustomFrameDepthDataset.get_normalization_transform(
    #     temp_dataset
    # )

    # transform = transforms.Compose(
    #     [
    #         transform,
    #         normalize_transform,
    #     ]
    # )

    # train_dataset = CustomFrameDepthDataset(
    #     os.path.join(input_path, "train"),
    #     transform=transform,
    # )
    # else:

    # train_dataset = CustomFrameDepthDataset(
    #     # os.path.join(input_path, "train"),
    #     input_path,
    #     transform=transform,
    #     start_idx=0,
    #     end_idx=500,
    #     # max_frames=1358,
    # )
    # train_dataset = Dataset(
    #     input_paths=train_imgs,
    #     target_paths=train_maps,
    #     transform_input=transform_input,
    #     transform_target=transform_target,
    #     hflip=True,
    #     vflip=True,
    #     affine=False,
    # )

    # val_dataset = Dataset(
    #     input_paths=val_imgs,
    #     target_paths=val_maps,
    #     transform_input=transform_input,
    #     transform_target=transform_target,
    # )

    # val_dataset = CustomFrameDepthDataset(
    #     # os.path.join(input_path, "val"),
    #     input_path,
    #     transform=transform,
    #     start_idx=0,
    #     end_idx=500,
    #     # max_frames=1358,
    # )
    # train_dataloader = data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     drop_last=True,
    #     shuffle=True,
    #     num_workers=4,
    # )
    # logger.info("Train dataloader created and processed")

    # val_dataloader = data.DataLoader(
    #     val_dataset,
    #     batch_size=batch_size,
    #     drop_last=False,
    #     shuffle=False,
    #     num_workers=4,
    # )
    # logger.info("Validation dataloader created and processed")

    # Set up model
    model = DepthEstimationModule(
        f"depth-anything/Depth-Anything-V2-{model_size}-hf",
        lr=lr,
        use_scheduler=use_scheduler,
        warmup_steps=warmup_steps,
    )

    custom_model_name = f"pretrained_l{lr}_e{epochs}_b{batch_size}_m{model_size}_s{1 if use_scheduler else 0}_w{warmup_steps if warmup_steps else 0}"

    # Set up TensorBoard logger
    tb_logger = TensorBoardLogger(
        "tb_logs",
        name="depth_estimation",
        version=custom_model_name,
    )

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=tb_logger,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
        accumulate_grad_batches=2,
        log_every_n_steps=1,
    )

    # Train the model
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Save the fine-tuned model
    custom_output_path = os.path.join(output_path, custom_model_name)
    model.model.save_pretrained(custom_output_path)
    logger.info("Saved fine-tuned model to %s", custom_output_path)


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
        type=bool,
        default=False,
        help="Learning Rate Scheduler",
    )
    parser.add_argument(
        "-w",
        "--warmup_steps",
        type=int,
        default=500,
        help="Number of warmup steps",
    )

    args = parser.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    main(
        args.input_dir,
        args.output_dir,
        args.batch_size,
        args.learning_rate,
        args.model_size,
        args.epochs,
        args.use_scheduler,
        args.warmup_steps,
    )
