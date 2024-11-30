"""Script for testing pretrained video depth estimation models on different datasets."""

import hydra
import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from data_processing import dataset
import lightning_model
import lightning_model_combined

# Set float32 matrix multiplication precision to 'high' for better performance
torch.set_float32_matmul_precision("high")


@hydra.main(
    config_path="configs",
    config_name="test_config",  # You'll need to create a new test config file
    version_base=None,
)
def main(
    args: DictConfig,
):
    pl.seed_everything(42)

    # Select appropriate datamodule based on config
    if args.dataset.ds_type == "simcol":
        data_module = dataset.SimColDataModule(**args.dataset)
        model_args = dict(args.model)
        model_args["max_depth"] = args.model.simcol_max_depth
    elif args.dataset.ds_type == "c3vd":
        data_module = dataset.C3VDDataModule(**args.dataset)
        model_args = dict(args.model)
        model_args["max_depth"] = args.model.c3vd_max_depth
    elif args.dataset.ds_type == "combined":
        data_module = dataset.CombinedDataModule(**args.dataset)
        model_args = dict(args.model)
    else:
        raise ValueError(f"Unknown dataset ds_type: {args.dataset.ds_type}")

    # Load pretrained model from checkpoint
    if args.dataset.ds_type != "combined":
        model = lightning_model.DepthAnythingV2Module.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            encoder=args.model.encoder,
            min_depth=args.model.min_depth,
            max_depth=model_args["max_depth"],
            lr=args.model.lr,
        )
    else:
        model = lightning_model_combined.DepthAnythingV2Module.load_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            **model_args,
        )

    # Set up logger
    experiment_id = f"test_m{args.model.encoder}_d{args.dataset.ds_type}"
    logger = False
    if args.logger:
        logger = WandbLogger(
            project=f"depth-any-endoscopy-{args.dataset.ds_type}",
            name=experiment_id,
            save_dir="~/home/public/avaishna/Endoscopy-3D-Modeling/",
            offline=False,
        )

    # Set up trainer for testing only
    trainer = pl.Trainer(
        **args.trainer,
        logger=logger,
        precision="16-mixed",
        inference_mode=True,
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

    # Example usage:
    # python test_lightning.py dataset=c3vd checkpoint_path="/home/public/avaishna/Endoscopy-3D-Modeling/checkpoints/mvitl_l1e-05_b20_e5_dc3vd/depth-any-endoscopy-epoch04-val_loss23.94.ckpt" trainer.devices=[0] ++dataset.batch_size=16
