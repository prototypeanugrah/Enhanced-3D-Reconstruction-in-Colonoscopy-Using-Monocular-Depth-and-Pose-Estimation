"""Main script for video depth estimation."""

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
):

    pl.seed_everything(42)

    # Select appropriate datamodule based on config
    if args.dataset.ds_type == "simcol":
        data_module = simcol.SimColDataModule(**args.dataset)
        model_args = dict(args.model)
        model_args["max_depth"] = args.model.simcol_max_depth
        del model_args["simcol_max_depth"]
        del model_args["c3vd_max_depth"]
    elif args.dataset.ds_type == "c3vd":
        data_module = c3vd.C3VDDataModule(**args.dataset)
        model_args = dict(args.model)
        model_args["max_depth"] = args.model.c3vd_max_depth
        del model_args["simcol_max_depth"]
        del model_args["c3vd_max_depth"]
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

    experiment_id = f"m{args.model.encoder}_l{args.model.lr}_b{args.dataset.batch_size}_e{args.trainer.max_epochs}_d{args.dataset.ds_type}_p{args.model.pct_start}"
    logger = False
    if args.logger:
        logger = WandbLogger(
            project=f"depth-any-endoscopy-{args.dataset.ds_type}",
            name=experiment_id,
            save_dir="~/home/public/avaishna/Endoscopy-3D-Modeling/",
            offline=False,
        )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{args.dataset.ds_type}/{experiment_id}",
        filename="depth-any-endoscopy-epoch{epoch:02d}-val_loss{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        verbose=True,
        min_delta=1e-4,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callback = [
        checkpoint_callback,
        # early_stopping,
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

    # Test the model
    if args.test:
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Loading best model from {best_model_path}")
            if args.dataset.ds_type != "combined":
                model = lightning_model.DepthAnythingV2Module.load_from_checkpoint(
                    checkpoint_path=best_model_path,
                    **model_args,
                )
            else:
                model = (
                    lightning_model_combined.DepthAnythingV2Module.load_from_checkpoint(
                        checkpoint_path=best_model_path,
                        **model_args,
                    )
                )

        # Run test set evaluation
        test_results = trainer.test(
            model=model,
            datamodule=data_module,
        )

        print("Test Results:", test_results[0])


if __name__ == "__main__":
    main()

    # Example script
    # python main_lightning.py ++dataset.batch_size=12 dataset=c3vd model=large ++trainer.devices=[1] ++model.lr=5e-2
