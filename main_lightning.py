"""
This script is used to train the DepthAnythingV2 model using PyTorch Lightning.
It supports training on SimCol, C3VD, or combined datasets. The script uses
Hydra for configuration management and Weights & Biases for logging.

Usage:
    python main_lightning.py \
    dataset.batch_size=12 \
    dataset.ds_type=c3vd \
    model.encoder=large \
    trainer.devices=[1] \
    model.encoder_lr=5e-6 \
    model.decoder_lr=5e-5 \
    trainer.max_epochs=1
    
Arguments:
    dataset.batch_size: Batch size for training.
    dataset.ds_type: Dataset type - simcol, c3vd, or combined.
    
    model.encoder: Encoder type - small, medium, large.
    trainer.devices: List of GPU devices to use.
    model.lr: Learning rate.
    trainer.max_epochs: Maximum number of epochs.

"""

import hydra
import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig

from data_processing import (
    simcol,
    c3vd,
    combined,
)
import lightning_model
import lightning_model_combined

# Set float32 matrix multiplication precision to 'high' for better performance
# on Tensor Cores
torch.set_float32_matmul_precision("high")


@hydra.main(
    config_path="configs",
    config_name="default",
    version_base=None,
)
def main(
    args: DictConfig,
) -> None:

    pl.seed_everything(42)

    # Select appropriate datamodule based on config

    # SimCol
    if args.dataset.ds_type == "simcol":
        data_module = simcol.SimColDataModule(**args.dataset)
        model_args = dict(args.model)
        model_args["max_depth"] = args.model.simcol_max_depth
        del model_args["simcol_max_depth"]
        del model_args["c3vd_max_depth"]

    # C3VD
    elif args.dataset.ds_type == "c3vd":
        data_module = c3vd.C3VDDataModule(**args.dataset)
        model_args = dict(args.model)
        model_args["max_depth"] = args.model.c3vd_max_depth
        del model_args["simcol_max_depth"]
        del model_args["c3vd_max_depth"]

    # Combined
    elif args.dataset.ds_type == "combined":
        data_module = combined.CombinedDataModule(**args.dataset)
        model_args = dict(args.model)

    else:
        raise ValueError(f"Unknown dataset ds_type: {args.dataset.ds_type}")

    # Set up model
    if args.dataset.ds_type != "combined":
        model = lightning_model.DepthAnythingV2Module(**model_args)
    else:
        model = lightning_model_combined.DepthAnythingV2Module(**model_args)

    experiment_id = (
        f"m{args.model.encoder}_el{args.model.encoder_lr}_"
        f"dl{args.model.decoder_lr}_b{args.dataset.batch_size}_"
        f"e{args.trainer.max_epochs}_d{args.dataset.ds_type}_"
        f"p{args.model.pct_start:.2f}"
    )

    logger = WandbLogger(
        project=f"depth-any-endoscopy-{args.dataset.ds_type}",
        name=experiment_id,
        save_dir="~/home/public/avaishna/Endoscopy-3D-Modeling/",
        offline=False,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{args.dataset.ds_type}/{experiment_id}",
        filename="depth_any_endoscopy_{epoch:02d}_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=40,
        mode="min",
        verbose=True,
        min_delta=1e-4,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callback = [
        checkpoint_callback,
        early_stopping,
    ]
    if logger:
        callback.append(lr_monitor)

    trainer = pl.Trainer(
        **args.trainer,
        logger=logger,
        callbacks=callback,
        # precision="32-true",
    )

    # Train the model
    trainer.fit(
        model,
        datamodule=data_module,
    )


if __name__ == "__main__":
    main()

