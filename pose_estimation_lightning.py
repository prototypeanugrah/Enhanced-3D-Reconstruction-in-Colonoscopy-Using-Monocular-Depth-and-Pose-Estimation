"""
This script is used to train the PoseEstimationNet model using PyTorch Lightning.
It supports training on the SimCol dataset for pose estimation between consecutive frames.

Usage:
    python pose_estimation_lightning.py \
    dataset.batch_size=32 \
    trainer.devices=[0] \
    model.lr=1e-4 \
    trainer.max_epochs=100

Arguments:
    dataset.batch_size: Batch size for training
    trainer.devices: List of GPU devices to use
    model.lr: Learning rate
    trainer.max_epochs: Maximum number of epochs
"""

from collections import defaultdict
import json
import os

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from omegaconf import DictConfig
import hydra
import lightning as pl
import numpy as np
import torch
import torchmetrics

from data_processing import pose_estimation
import pose_estimation_model

# Set float32 matrix multiplication precision
torch.set_float32_matmul_precision("high")


class ProcedureMetricCollector(pl.Callback):
    """
    ProcedureMetricCollector is a PyTorch Lightning callback that collects
    metrics for each procedure during testing.

    Args:
        pl (Callback): PyTorch Lightning callback
    """

    def __init__(self, dataset: pose_estimation.PoseDataset) -> None:
        """
        Initialize the ProcedureMetricCollector

        Args:
            dataset (pose_estimation.PoseDataset): Pose dataset for evaluation
        """
        super().__init__()
        self.dataset = dataset
        self.metrics_by_procedure = defaultdict(list)

        # Initialize metrics
        self.metrics = torchmetrics.MetricCollection(
            {
                "ate": torchmetrics.MeanMetric(),
                "rte": torchmetrics.MeanMetric(),
                "rote": torchmetrics.MeanMetric(),
            }
        )

    def on_test_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """
        Callback function to initialize the metrics at the start of the test
        epoch.

        Args:
            trainer (pl.Trainer): PyTorch Lightning trainer
            pl_module (pl.LightningModule): PyTorch Lightning module
        """
        # Move metrics to the correct device
        self.metrics = self.metrics.to(pl_module.device)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        """
        Callback function to update the metrics at the end of each test batch.

        Args:
            trainer (pl.Trainer): PyTorch Lightning trainer
            pl_module (pl.LightningModule): PyTorch Lightning module
            outputs (dict): Model outputs
            batch (dict): Batch data
            batch_idx (int): Batch index

        Raises:
            ValueError: If the outputs do not contain the expected keys
        """

        # Ensure outputs contain the expected keys
        if not all(
            key in outputs
            for key in [
                "ate",
                "rte",
                "rote",
            ]
        ):
            raise ValueError("Missing expected keys in outputs")

        # Convert numpy values to torch tensors if needed
        ate = (
            outputs["ate"]
            if torch.is_tensor(outputs["ate"])
            else torch.tensor(outputs["ate"])
        )
        rte = (
            outputs["rte"]
            if torch.is_tensor(outputs["rte"])
            else torch.tensor(outputs["rte"])
        )
        rote = (
            outputs["rote"]
            if torch.is_tensor(outputs["rote"])
            else torch.tensor(outputs["rote"])
        )

        # Update metrics with the current batch
        self.metrics["ate"].update(ate.to(pl_module.device))
        self.metrics["rte"].update(rte.to(pl_module.device))
        self.metrics["rote"].update(rote.to(pl_module.device))
        batch_metrics = self.metrics.compute()

        # Get the file paths for this batch
        batch_size = len(batch["input"])
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

            self.metrics_by_procedure[procedure_full].append(
                batch_metrics
            )  # Use batch_metrics here

        self.metrics.reset()


@hydra.main(
    config_path="configs/pose_estimation",
    config_name="pose_estimation",
    version_base=None,
)
def main(args: DictConfig) -> None:
    pl.seed_everything(42)

    # Initialize data module
    data_module = pose_estimation.PoseDataModule(**args.dataset)

    # Initialize model
    model = pose_estimation_model.PoseEstimationModule(**args.model)

    # Create unique experiment ID
    experiment_id = (
        f"pose_est_lr{args.model.lr}_"
        f"b{args.dataset.batch_size}_"
        f"e{args.trainer.max_epochs}_"
        f"wd{args.model.weight_decay}_"
        f"beta{args.model.beta}"
    )

    # Initialize WandB logger
    logger = WandbLogger(
        project="depth-any-endoscopy-pose",
        name=experiment_id,
        save_dir="~/home/public/avaishna/Endoscopy-3D-Modeling/",
        offline=False,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/pose_estimation/{experiment_id}",
        filename="pose_estimation_{epoch:02d}_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        verbose=True,
        min_delta=1e-6,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [
        checkpoint_callback,
        # early_stopping,
        lr_monitor,
    ]

    # Initialize metric collector
    metric_collector = ProcedureMetricCollector(data_module.test_dataset)

    # Initialize trainer
    trainer = pl.Trainer(
        **args.trainer,
        logger=logger,
        callbacks=(
            callbacks + ([metric_collector] if args.dataset.ds_type == "simcol" else [])
        ),
    )

    # Train the model
    trainer.fit(
        model,
        datamodule=data_module,
    )

    ######################
    #### Testing ####
    ######################

    # # Initialize metric collector
    # metric_collector = ProcedureMetricCollector(data_module.test_dataset)

    # Set up trainer for testing only
    # trainer = pl.Trainer(
    #     **args.trainer,
    #     logger=logger,
    #     inference_mode=True,
    #     callbacks=[metric_collector] if args.dataset.ds_type == "simcol" else None,
    # )

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
                    m["ate"].cpu(),
                    m["rte"].cpu(),
                    m["rote"].cpu(),
                ]
                for m in metrics_list
            ]
        )
        mean_metrics = np.mean(metrics_array, axis=0)

        print(f"\nProcessing {procedure}")
        print(f"Mean Absolute Translational Error (ATE): {mean_metrics[0]:.6f}")
        print(f"Mean Relative Translational Error (RTE): {mean_metrics[1]:.6f}")
        print(f"Mean Rotation Error (ROT): {mean_metrics[2]:.6f}")
        print("-" * 50)

        # Store for overall statistics
        for i, metric_name in enumerate(["ate", "rte", "rote"]):
            all_metrics[metric_name].append(mean_metrics[i])

    # Print overall statistics
    print("\nOverall Results:")
    for metric_name in ["ate", "rte", "rote"]:
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
        os.path.join(results_dir, "pose_estimation_results_ocl_e50.json"),
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
