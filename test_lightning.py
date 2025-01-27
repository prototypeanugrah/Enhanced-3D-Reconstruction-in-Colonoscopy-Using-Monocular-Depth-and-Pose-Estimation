"""Script for testing pretrained video depth estimation models on different datasets."""

from collections import defaultdict
import hydra
import os
import gc
import json
import lightning as pl
import numpy as np
import torch
import torchmetrics
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


class ProcedureMetricCollector(pl.Callback):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.metrics_by_procedure = defaultdict(list)

        # Initialize metrics
        self.metrics = torchmetrics.MetricCollection(
            {
                "d1": torchmetrics.MeanMetric(),
                "abs_rel": torchmetrics.MeanMetric(),
                "rmse": torchmetrics.MeanMetric(),
                "l1": torchmetrics.MeanMetric(),
            }
        )

    def on_test_epoch_start(self, trainer, pl_module):
        # Move metrics to the correct device
        device = pl_module.device
        self.metrics = self.metrics.to(device)

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ):

        # print(outputs)
        # exit(0)

        # Ensure outputs contain the expected keys
        if not all(
            key in outputs
            for key in [
                "d1",
                "abs_rel",
                "rmse",
                "l1",
            ]
        ):
            raise ValueError("Missing expected keys in outputs")

        # Update metrics with the current batch
        self.metrics["l1"].update(outputs["l1"].to(pl_module.device))
        self.metrics["abs_rel"].update(outputs["abs_rel"].to(pl_module.device))
        self.metrics["d1"].update(outputs["d1"].to(pl_module.device))
        self.metrics["rmse"].update(outputs["rmse"].to(pl_module.device))
        # self.metrics.update(
        #     d1=outputs["d1"],
        #     abs_rel=outputs["abs_rel"],
        #     rmse=outputs["rmse"],
        #     l1=outputs["l1"],
        # )

        # Get metrics from outputs
        # batch_metrics = {
        #     "l1": pl_module.metric["l1"].compute().item(),
        #     "abs_rel": pl_module.metric["abs_rel"].compute().item(),
        #     "d1": pl_module.metric["d1"].compute().item(),
        #     "rmse": pl_module.metric["rmse"].compute().item(),
        # }
        batch_metrics = self.metrics.compute()

        # Get the file paths for this batch
        batch_size = len(batch["image"])
        for i in range(batch_size):
            # Initialize variables
            colon_type = None
            procedure = None

            # Get dataset path
            dataset_path = str(batch["dataset"][i])
            # Get frame ID
            frame_id = str(batch["id"][i])

            # Extract procedure info from dataset path and frame ID
            parts = dataset_path.split("/")
            for part in parts:
                if part.startswith("SyntheticColon_"):
                    colon_type = part

            # Get procedure from frame ID (assuming it contains the procedure info)
            # Handle all procedure types (S, B, O)
            if "S" in frame_id:
                procedure = f"Frames_S{frame_id.split('S')[1].split('_')[0]}"
            elif "B" in frame_id:
                procedure = f"Frames_B{frame_id.split('B')[1].split('_')[0]}"
            elif "O" in frame_id:
                procedure = f"Frames_O{frame_id.split('O')[1].split('_')[0]}"

            if colon_type is None or procedure is None:
                continue

            procedure_full = f"{colon_type}/{procedure}"

            # metrics = {
            #     "l1": pl_module.metric["l1"].compute().item(),
            #     "abs_rel": pl_module.metric["abs_rel"].compute().item(),
            #     "d1": pl_module.metric["d1"].compute().item(),
            #     "rmse": pl_module.metric["rmse"].compute().item(),
            # }

            self.metrics_by_procedure[procedure_full].append(
                batch_metrics
            )  # Use batch_metrics here

        self.metrics.reset()


def load_model_weights(model, checkpoint_path, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Assuming the checkpoint is a direct state_dict
        state_dict = checkpoint

        # Fix the key prefixes
        new_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith("model."):
                new_key = f"model.{key}"
            else:
                new_key = key
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict)
        del state_dict, checkpoint, new_state_dict
    return model


def load_checkpoint_with_fallback(checkpoint_path, map_location=None):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    print("Checkpoint keys before modification:", checkpoint.keys())  # Debugging line
    if "pytorch-lightning_version" not in checkpoint:
        print("Adding missing 'pytorch-lightning_version' key")  # Debugging line
        checkpoint["pytorch-lightning_version"] = (
            "2.5.0.post0"  # or set to a default version
        )
    return checkpoint


@hydra.main(
    config_path="configs",
    config_name="test_config",
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
        model = lightning_model.DepthAnythingV2Module(
            checkpoint_path=checkpoint_path,
            encoder=args.model.encoder,
            min_depth=args.model.min_depth,
            max_depth=model_args["max_depth"],
            encoder_lr=args.model.encoder_lr,
            decoder_lr=args.model.decoder_lr,
            cycle_momentum=args.model.cycle_momentum,
            div_factor=args.model.div_factor,
            pct_start=args.model.pct_start,
            map_location="cpu",
            # _load_checkpoint=load_checkpoint_with_fallback,
        )
    else:
        model = lightning_model_combined.DepthAnythingV2Module.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location="cpu",
            **model_args,
        )

    model = load_model_weights(
        model,
        checkpoint_path,
        map_location="cpu",
    )

    # Set up logger
    experiment_id = (
        "test_"
        f"m{args.model.encoder}_el{args.model.encoder_lr}_"
        f"dl{args.model.decoder_lr}_b{args.dataset.batch_size}_"
        f"e{args.trainer.max_epochs}_d{args.dataset.ds_type}_"
        f"p{args.model.pct_start:.2f}_div{args.model.div_factor}"
    )
    logger = False
    if args.logger:
        logger = WandbLogger(
            project=f"depth-any-endoscopy-{args.dataset.ds_type}",
            name=experiment_id,
            save_dir="~/home/public/avaishna/Endoscopy-3D-Modeling/",
            offline=False,
        )

    metric_collector = ProcedureMetricCollector(data_module.test_dataset)

    # Set up trainer for testing only
    trainer = pl.Trainer(
        **args.trainer,
        logger=logger,
        inference_mode=True,
        callbacks=[metric_collector] if args.dataset.ds_type == "simcol" else None,
    )

    # Run test set evaluation
    trainer.test(
        model=model,
        datamodule=data_module,
    )

    # Calculate and print metrics for each procedure
    print("\nResults by Procedure:")
    all_metrics = defaultdict(list)

    for procedure, metrics_list in metric_collector.metrics_by_procedure.items():
        metrics_array = np.array(
            [
                [
                    m["l1"].cpu(),
                    m["abs_rel"].cpu(),
                    m["d1"].cpu(),
                    m["rmse"].cpu(),
                ]
                for m in metrics_list
            ]
        )
        mean_metrics = np.mean(metrics_array, axis=0)

        print(f"\nProcessing {procedure}")
        print(f"Mean L1 error: {mean_metrics[0]:.6f}")
        print(f"Mean AbsRel error: {mean_metrics[1]:.6f}")
        print(f"Mean δ<1.1: {mean_metrics[2]:.6f}")
        print(f"Mean RMSE: {mean_metrics[3]:.6f}")
        print("-" * 50)

        # Store for overall statistics
        for i, metric_name in enumerate(["l1", "abs_rel", "d1", "rmse"]):
            all_metrics[metric_name].append(mean_metrics[i])

    # Print overall statistics
    print("\nOverall Results:")
    for metric_name in ["l1", "abs_rel", "d1", "rmse"]:
        mean_val = np.mean(all_metrics[metric_name])
        std_val = np.std(all_metrics[metric_name])
        print(f"Overall {metric_name}: {mean_val:.6f} ± {std_val:.6f}")

    results = {
        "metrics_by_procedure": dict(metric_collector.metrics_by_procedure),
        "overall_metrics": {
            metric: {"mean": np.mean(values), "std": np.std(values)}
            for metric, values in all_metrics.items()
        },
    }

    results_dir = "test_lightning_results"
    os.makedirs(results_dir, exist_ok=True)

    with open(
        os.path.join(results_dir, "simcol_results_bs.json"),
        "w",
        encoding="utf-8",
    ) as f:
        # Convert numpy values to float for JSON serialization
        json_results = {
            "metrics_by_procedure": {
                k: [{m_k: float(m_v) for m_k, m_v in m.items()} for m in v]
                for k, v in results["metrics_by_procedure"].items()
            },
            "overall_metrics": {
                k: {sk: float(sv) for sk, sv in v.items()}
                for k, v in results["overall_metrics"].items()
            },
        }
        json.dump(json_results, f, indent=4)


if __name__ == "__main__":
    main()

    # Example usage:

    # Before running the script, set the CHECKPOINT_PATH to the actual checkpoint path - export CHECKPOINT_PATH="./checkpoints/combined/mvitl_el5e-06_dl5e-05_b6_e30_dcombined_p0.05/depth_any_endoscopy_epoch=17_val_loss=0.06.ckpt"

    # Simcol - "./checkpoints/simcol/mvitl_el5e-06_dl5e-05_b6_e30_dsimcol_p0.05/depth_any_endoscopy_epoch=29_val_loss=0.02.ckpt"
    # python test_lightning.py dataset=simcol trainer.devices=[0] ++dataset.batch_size=32

    # C3VD - "./checkpoints/c3vd/mvitl_el5e-06_dl5e-05_b6_e20_dc3vd_p0.05/depth_any_endoscopy_epochepoch=13_val_lossval_loss=0.10.ckpt"
    # python test_lightning.py dataset=c3vd trainer.devices=[0] ++dataset.batch_size=16

    # Combined - "./checkpoints/combined/mvitl_el5e-06_dl5e-05_b6_e30_dcombined_p0.05/depth_any_endoscopy_epoch=17_val_loss=0.06.ckpt"
