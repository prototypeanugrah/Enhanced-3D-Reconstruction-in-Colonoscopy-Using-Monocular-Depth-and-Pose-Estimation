"""Script for testing pretrained video depth estimation models on different datasets."""

import hydra
import os
import gc
import lightning as pl
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from data_processing import (
    simcol,
    c3vd,
    combined,
)
import lightning_model
import lightning_model_combined

# Set float32 matrix multiplication precision to 'high' for better performance
torch.set_float32_matmul_precision("high")
# torch.cuda.set_device(5)


@hydra.main(
    config_path="configs",
    config_name="test_config",  # You'll need to create a new test config file
    version_base=None,
)
def main(
    args: DictConfig,
):

    # Force garbage collection and clear CUDA cache
    gc.collect()
    torch.cuda.empty_cache()

    # Get checkpoint path from environment variable
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if checkpoint_path is None:
        raise ValueError("Please set the CHECKPOINT_PATH environment variable")

    pl.seed_everything(42)

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

    # Load pretrained model from checkpoint
    if args.dataset.ds_type != "combined":
        model = lightning_model.DepthAnythingV2Module.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            encoder=args.model.encoder,
            min_depth=args.model.min_depth,
            max_depth=model_args["max_depth"],
            lr=args.model.encoder_lr,
            map_location="cpu",
        )
    else:
        model = lightning_model_combined.DepthAnythingV2Module.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cpu",
            **model_args,
        )

    # Set up logger
    experiment_id = (
        "test_"
        f"m{args.model.encoder}_el{args.model.encoder_lr}_"
        f"dl{args.model.decoder_lr}_b{args.dataset.batch_size}_"
        f"e{args.trainer.max_epochs}_d{args.dataset.ds_type}_"
        f"p{args.model.pct_start:.2f}"
    )
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
        inference_mode=True,
    )

    # Run test set evaluation
    trainer.test(
        model=model,
        datamodule=data_module,
    )


if __name__ == "__main__":
    main()

    # Example usage:
    # python test_lightning.py dataset=c3vd checkpoint_path="/home/public/avaishna/Endoscopy-3D-Modeling/checkpoints/mvitl_l1e-05_b20_e5_dc3vd/depth-any-endoscopy-epoch04-val_loss23.94.ckpt" trainer.devices=[0] ++dataset.batch_size=16
    # python test_lightning.py dataset=simcol ++dataset.batch_size=1
    # Before running the script, set the CHECKPOINT_PATH to the actual checkpoint path -
    # Simcol - /home/public/avaishna/Endoscopy-3D-Modeling/checkpoints/simcol/mvitl_el5e-06_dl5e-05_b6_e30_dsimcol_p0.05/depth_any_endoscopy_epoch=29_val_loss=0.02.ckpt
    # C3VD -
