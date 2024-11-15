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

from data_processing import dataset
import lightning_model

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

    # Data module
    # data_module = dataset.SimColDataModule(**args.dataset)
    data_module = dataset.CombinedDataModule(**args.dataset)

    # Set up model
    model = lightning_model.DepthAnythingV2Module(**args.model)

    experiment_id = f"m{args.model.encoder}_l{args.model.lr}_b{args.dataset.batch_size}_e{args.trainer.max_epochs}"
    logger = False
    if args.logger:
        logger = WandbLogger(
            project="depth-any-endoscopy-combined",
            name=experiment_id,
            save_dir="~/home/public/avaishna/Endoscopy-3D-Modeling/",
            offline=False,
        )
        experiment_id = logger.experiment.id

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{experiment_id}",
        filename="depth-any-endoscopy-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callback = [checkpoint_callback, early_stopping]
    if logger:
        callback.append(lr_monitor)

    trainer = pl.Trainer(
        **args.trainer,
        logger=logger,
        callbacks=callback,
        log_every_n_steps=100,
        # precision="32-true",
        precision="16-mixed",
        val_check_interval=0.5,
    )

    # Train the model
    trainer.fit(
        model,
        datamodule=data_module,
    )

    # Test the model
    if args.test:
        # Load best model
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            print(f"Loading best model from {best_model_path}")
            model = lightning_model.DepthAnythingV2Module.load_from_checkpoint(
                checkpoint_path=best_model_path
            )

        # Run test set evaluation
        test_results = trainer.test(
            model=model,
            datamodule=data_module,
        )

        # Log test results
        if logger:
            metrics = {f"test_{k}": v for k, v in test_results[0].items()}
            logger.log_metrics(metrics)

        print("Test Results:", test_results)


if __name__ == "__main__":
    main()

    # Example script
    # python main_lightning.py ++dataset.batch_size=12 model=large ++trainer.devices=[1] ++model.lr=5e-2
