"Module for training/fine-tuning the DepthAnythingV2 model"

import argparse
import logging
import os

from custom_dataset import CustomVideoDepthDataset
from lightning_model import DepthEstimationModule
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms

import torch
import pytorch_lightning as pl

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(
    input_path: str,
    output_path: str,
    batch_size: int,
    lr: float,
    model_size: str,
    epochs: int,
):
    # Set up data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((384, 384)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = CustomVideoDepthDataset(
        os.path.join(input_path, "train"),
        transform=transform,
    )
    val_dataset = CustomVideoDepthDataset(
        os.path.join(input_path, "val"),
        transform=transform,
    )
    # test_dataset = CustomVideoDepthDataset(
    #     os.path.join(input_path, "test"),
    #     transform=transform,
    # )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Set up model
    model = DepthEstimationModule(
        f"depth-anything/Depth-Anything-V2-{model_size}-hf",
        lr=lr,
    )

    # Set up TensorBoard logger
    tb_logger = TensorBoardLogger("tb_logs", name="depth_estimation")

    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=tb_logger,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="val_loss")],
    )

    # Train the model
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
    )

    # Save the fine-tuned model
    custom_model_name = f"pretrained_l{lr}_e{epochs}_b{batch_size}"
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
        choices=["small", "medium", "large"],
        help="Size of the DepthAnythingV2 model to use",
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
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
    )
