"""Main script for video depth estimation."""

import argparse
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pathlib import Path
import copy

# from omegaconf import DictConfig
# import hydra
import lightning as pl
import torch
import yaml

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


def load_dataset_config(config_path):

    dataset_params = config_path["dataset"]
    ds_type = dataset_params["ds_type"]

    if ds_type == "simcol":
        dataset_params = {
            "data_dir": dataset_params["simcol_data_dir"],
            "train_list": dataset_params["simcol_train_list"],
            "val_list": dataset_params["simcol_val_list"],
            "test_list": dataset_params["simcol_test_list"],
        }
    elif ds_type == "c3vd":
        dataset_params = {
            "data_dir": dataset_params["c3vd_data_dir"],
            "train_list": dataset_params["c3vd_train_list"],
            "val_list": dataset_params["c3vd_val_list"],
            # "test_list": None,  # Add test_list if required for C3VD
        }
    elif ds_type == "combined":
        pass
        # dataset_params = {
        #     # SimCol
        #     "simcol_data_dir": dataset_params["simcol_data_dir"],
        #     "simcol_train_list": dataset_params["simcol_train_list"],
        #     "simcol_val_list": dataset_params["simcol_val_list"],
        #     "simcol_test_list": dataset_params["simcol_test_list"],
        #     # C3VD
        #     "c3vd_data_dir": dataset_params["c3vd_data_dir"],
        #     "c3vd_train_list": dataset_params["c3vd_train_list"],
        #     "c3vd_val_list": dataset_params["c3vd_val_list"],
        # }
    else:
        raise ValueError(f"Unsupported ds_type: {ds_type}")

    # Include common parameters
    dataset_params.update(
        {
            "batch_size": config_path["dataset"]["batch_size"],
            "num_workers": config_path["dataset"]["num_workers"],
            "size": config_path["dataset"]["size"],
            "ds_type": ds_type,
        }
    )

    return dataset_params


def update_nested_dict(d, key_path, value):
    """Update nested dictionary using dot notation key path"""
    keys = key_path.split(".")
    current = d
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def load_and_override_config(config_path, overrides):
    # Load base config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Convert list-style sections to dictionaries
    # processed_config = {}
    # for section_name, section_list in config.items():
    #     processed_config[section_name] = convert_list_to_dict(section_list)

    processed_config = {
        section_name: dict(item for d in section_list for item in d.items())
        for section_name, section_list in config.items()
    }

    # Apply overrides
    for override in overrides:
        if "=" not in override:
            continue
        key_path, value = override.split("=")
        section, key = key_path.split(".")

        # Try to convert value to appropriate type
        try:
            # Only evaluate if it looks like a list or number
            if (
                value.startswith("[")
                or value.startswith("(")
                or value.replace(".", "", 1).isdigit()
            ):
                value = eval(value)
            # Otherwise keep as string
        except:
            pass  # Keep as string if eval fails

        if section in processed_config:
            processed_config[section][key] = value

    return processed_config


def save_used_config(
    config,
    experiment_id,
):
    """Save the actual config used for training"""
    output_dir = Path(f"configs/experiments/{experiment_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert back to list format for saving
    list_config = {}
    for section_name, section_dict in config.items():
        list_config[section_name] = [{k: v} for k, v in section_dict.items()]

    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(list_config, f, default_flow_style=False)


def convert_list_to_dict(config):
    """Convert list-style config to dictionary"""
    result = {}
    for section in config:
        for key, value in section.items():
            result[key] = value
    return result


# @hydra.main(
#     config_path="configs",
#     config_name="default",
#     version_base=None,
# )


def parse_args():
    parser = argparse.ArgumentParser(description="Depth Estimation Training")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "-o",
        "--override",
        nargs="*",
        default=[],
        help="Override config values (e.g., dataset.batch_size=32 model.lr=0.001)",
    )
    return parser.parse_args()


def main(
    # args: DictConfig,
    config,
):

    pl.seed_everything(42)

    # Extract parameters from config
    dataset_params = load_dataset_config(config)
    model_params = config["model"]
    trainer_params = config["trainer"]

    # Select appropriate datamodule based on config
    # if dataset_params["ds_type"] == "simcol":
    #     data_module = simcol.SimColDataModule(dataset_params)
    #     # model_args = dict(args.model)
    #     model_params = copy.deepcopy(model_params)
    #     # model_args["max_depth"] = args.model.simcol_max_depth
    #     # del model_args["simcol_max_depth"]
    #     # del model_args["c3vd_max_depth"]
    #     model_params["max_depth"] = model_params["simcol_max_depth"]
    #     del model_params["simcol_max_depth"]
    #     del model_params["c3vd_max_depth"]
    # elif dataset_params["ds_type"] == "c3vd":
    #     data_module = c3vd.C3VDDataModule(dataset_params)
    #     # model_args = dict(args.model)
    #     model_params = copy.deepcopy(model_params)
    #     # model_args["max_depth"] = args.model.c3vd_max_depth
    #     # del model_args["simcol_max_depth"]
    #     # del model_args["c3vd_max_depth"]
    #     model_params["max_depth"] = model_params["c3vd_max_depth"]
    #     del model_params["simcol_max_depth"]
    #     del model_params["c3vd_max_depth"]
    if dataset_params["ds_type"] == "simcol":
        data_module = simcol.SimColDataModule(dataset_params)
        model_params = {
            k: v
            for k, v in model_params.items()
            if k not in ("simcol_max_depth", "c3vd_max_depth")
        }
        model_params["max_depth"] = config["model"]["simcol_max_depth"]
    elif dataset_params["ds_type"] == "c3vd":
        data_module = c3vd.C3VDDataModule(dataset_params)
        model_params = {
            k: v
            for k, v in model_params.items()
            if k not in ("simcol_max_depth", "c3vd_max_depth")
        }
        model_params["max_depth"] = config["model"]["c3vd_max_depth"]
    elif dataset_params["ds_type"] == "combined":
        data_module = combined.CombinedDataModule(dataset_params)
        # model_args = dict(args.model)
        # model_params = copy.deepcopy(model_params)
    else:
        raise ValueError(f"Unknown dataset ds_type: {dataset_params['ds_type']}")

    # Set up model
    if dataset_params["ds_type"] != "combined":
        model = lightning_model.DepthAnythingV2Module(**model_params)
    else:
        model = lightning_model_combined.DepthAnythingV2Module(**model_params)

    # experiment_id = f"m{args.model.encoder}_l{args.model.lr}_b{dataset_params['batch_size']}_e{args.trainer.max_epochs}_d{dataset_params['ds_type']}_p{args.model.pct_start}"
    experiment_id = f"m{model_params['encoder']}_b{dataset_params['batch_size']}_e{trainer_params['max_epochs']}_d{dataset_params['ds_type']}_p{model_params['pct_start']}"

    # Save the actual config used
    save_used_config(config, experiment_id)

    logger = WandbLogger(
        project=f"depth-any-endoscopy-{dataset_params['ds_type']}",
        name=experiment_id,
        save_dir="~/home/public/avaishna/Endoscopy-3D-Modeling/",
        offline=False,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{dataset_params['ds_type']}/{experiment_id}",
        filename="depth_any_endoscopy_epoch{epoch:02d}_val_loss{val_loss:.2f}",
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
        # **args.trainer,
        **trainer_params,
        logger=logger,
        callbacks=callback,
        # precision="32-true",
    )

    torch.cuda.empty_cache()

    # Train the model
    trainer.fit(
        model,
        datamodule=data_module,
    )


if __name__ == "__main__":

    args = parse_args()
    config = load_and_override_config(
        args.config,
        args.override,
    )

    # config_path = "./configs/default.yaml"
    # h_params = load_dataset_config(config_path)

    main(config)

    # Example script
    # python main_lightning.py ++dataset.batch_size=12 dataset=c3vd model=large ++trainer.devices=[1] ++model.lr=5e-2
