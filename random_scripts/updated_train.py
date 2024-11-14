"Module for training/fine-tuning the DepthAnythingV2 model"

import argparse
import logging
import os

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate import DistributedDataParallelKwargs
from torch import nn
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import transformers

from data_processing import dataloader
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from eval import evaluation
from training import training_utils
from utils import utils

# Set float32 matrix multiplication precision to 'high' for better performance
# on Tensor Cores
torch.set_float32_matmul_precision("high")

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("accelerate").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
os.environ["HF_HOME"] = "~/home/public/avaishna/.cache"


def train(
    model: DepthAnythingV2,
    optim: torch.optim.Optimizer,
    scheduler: transformers.get_polynomial_decay_schedule_with_warmup,
    criterion: nn.MSELoss,
    dataloader: torch.utils.data.DataLoader,
    epoch: int,
    writer: SummaryWriter,
    accelerator: Accelerator,
    max_depth: float,
) -> tuple:
    """
    Function to train the model on the training set.

    Args:
        model (DepthAnythingV2): Model to train.
        optim (torch.optim.Optimizer): Optimizer.
        scheduler (transformers.get_polynomial_decay_schedule_with_warmup): Scheduler.
        criterion (nn.MSELoss): Loss function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training set.
        epoch (int): Current epoch.
        writer (SummaryWriter): TensorBoard writer.
        accelerator (Accelerator): Accelerator object.
        max_depth (float): Maximum depth value.

    Returns:
        tuple: Training loss and results.
    """

    # Training
    model.train()
    train_loss = 0.0
    results = {"d1": 0, "abs_rel": 0, "rmse": 0}
    with tqdm(
        dataloader,
        desc=f"Training Epoch {epoch+1}",
        disable=not accelerator.is_local_main_process,
        leave=False,
    ) as pbar:
        for batch_idx, sample in enumerate(pbar):
            optim.zero_grad()

            img, depth = sample

            pred = model(img)

            assert (
                pred.shape == depth.shape
            ), f"Shape mismatch. Target: {depth.shape}, Output: {pred.shape}"

            # mask
            mask = (depth <= max_depth) & (depth >= 0.001)

            loss = criterion(pred[mask], depth[mask])

            metrics = evaluation.compute_errors(
                pred[mask],
                depth[mask],
            )

            for k in results.keys():
                results[k] += metrics[k]

            accelerator.backward(loss)

            # Add gradient clipping
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()
            scheduler.step()

            train_loss += loss.detach()

            # Update progress bar with current loss
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if batch_idx % 100 == 0:
                writer.add_scalar(
                    "Train/Loss",
                    loss.item(),
                    epoch * len(dataloader) + batch_idx,
                )

                # Add gradient norm logging
                total_norm = 0.0
                for _, p in model.named_parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2

                total_norm = total_norm**0.5
                writer.add_scalar(
                    "Train/GradientNorm",
                    total_norm,
                    epoch * len(dataloader) + batch_idx,
                )

    train_loss /= len(dataloader)
    train_loss = accelerator.reduce(train_loss, reduction="mean").item()

    for k in results.keys():
        results[k] = results[k] / len(dataloader)
        results[k] = round(
            accelerator.reduce(
                results[k],
                reduction="mean",
            ).item(),
            3,
        )
        writer.add_scalar(f"Train/{k}", results[k], epoch)

    return (
        train_loss,
        results,
    )


def test(
    model: DepthAnythingV2,
    criterion: nn.MSELoss,
    dataloader: torch.utils.data.DataLoader,
    epoch: int,
    writer: SummaryWriter,
    accelerator: Accelerator,
    max_depth: float,
) -> tuple:
    """
    Function to test the model on the validation set.

    Args:
        model (DepthAnythingV2): Model to test.
        criterion (nn.MSELoss): Loss function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
        epoch (int): Current epoch.
        writer (SummaryWriter): TensorBoard writer.
        accelerator (Accelerator): Accelerator object.
        max_depth (float): Maximum depth value.

    Returns:
        tuple: Tuple containing the loss and evaluation results.
    """

    # Validation
    model.eval()
    val_loss = 0.0
    results = {"d1": 0, "abs_rel": 0, "rmse": 0}
    with tqdm(
        dataloader,
        desc=f"Validating Epoch {epoch+1}",
        disable=not accelerator.is_local_main_process,
        leave=False,
    ) as pbar:
        for batch_idx, sample in enumerate(pbar):

            img, depth = sample

            with torch.no_grad():
                pred = model(img)
                # evaluate on the original resolution
                pred = F.interpolate(
                    pred[:, None],
                    depth.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )
                pred = pred.squeeze(1)

                assert (
                    pred.shape == depth.shape
                ), f"Shape mismatch. Target: {depth.shape}, Output: {pred.shape}"

                valid_mask = (depth <= max_depth) & (depth >= 0.001)

                loss = criterion(
                    pred[valid_mask],
                    depth[valid_mask],
                )
                val_loss += loss.detach()

                # Update progress bar with current loss
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                cur_results = evaluation.compute_errors(
                    pred[valid_mask],
                    depth[valid_mask],
                )

                if batch_idx % 50 == 0:
                    writer.add_scalar(
                        "Val/Loss",
                        loss.item(),
                        epoch * len(dataloader) + batch_idx,
                    )

                for k in results.keys():
                    results[k] += cur_results[k]

    val_loss /= len(dataloader)
    val_loss = accelerator.reduce(val_loss, reduction="mean").item()

    for k in results.keys():
        results[k] = results[k] / len(dataloader)
        results[k] = round(
            accelerator.reduce(
                results[k],
                reduction="mean",
            ).item(),
            3,
        )
        writer.add_scalar(f"Val/{k}", results[k], epoch)

    return (
        val_loss,
        results,
    )


def main(
    input_path: str,
    output_path: str,
    batch_size: int,
    lr: float,
    model_size: str,
    epochs: int,
    logdir: str = None,
):

    val_vids = [
        os.path.join(input_path, "SyntheticColon_I/Frames_S4"),
        os.path.join(input_path, "SyntheticColon_I/Frames_S9"),
        os.path.join(input_path, "SyntheticColon_I/Frames_S14"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B4"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B9"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B14"),
    ]
    test_vids = [
        os.path.join(input_path, "SyntheticColon_I/Frames_S5"),
        os.path.join(input_path, "SyntheticColon_I/Frames_S10"),
        os.path.join(input_path, "SyntheticColon_I/Frames_S15"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B5"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B10"),
        os.path.join(input_path, "SyntheticColon_II/Frames_B15"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O1"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O2"),
        os.path.join(input_path, "SyntheticColon_III/Frames_O3"),
    ]

    # Get all subdirectories
    all_dirs = []
    for root, dirs, _ in os.walk(input_path):
        for dir in dirs:
            if dir.startswith("Frames_"):
                all_dirs.append(os.path.join(root, dir))

    # Create train_vids by excluding val_vids and test_vids
    train_vids = [
        dir for dir in all_dirs if dir not in val_vids and dir not in test_vids
    ]

    # Process the images and get the train and validation dataloaders
    train_depth, train_rgb, val_depth, val_rgb = utils.process_images(
        train_vids,
        val_vids,
        test_vids,
        input_path,
    )

    train_dataloader, val_dataloader = dataloader.get_dataloaders(
        train_depth,
        train_rgb,
        val_depth,
        val_rgb,
        batch_size=batch_size,
    )

    # Train the model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model_weights_path = f"./depth_anything_v2_vit{model_size}.pth"
    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }
    model_encoder = f"vit{model_size}"
    max_depth = 20.0

    set_seed(42)

    # Calculate warmup steps
    model_size_warmup_steps = {
        "s": 0.5,
        "b": 0.1,
        "l": 0.1,
    }

    warmup_epochs = model_size_warmup_steps.get(
        model_size, 0.5
    )  # Default to 0.5 if not found
    num_warmup_steps = int(warmup_epochs * len(train_dataloader))

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[ddp_kwargs],
    )

    model = DepthAnythingV2(
        **{
            **model_configs[model_encoder],
        }
    )

    # Load pretrained weights and explicitly unfreeze all parameters
    model.load_state_dict(
        {k: v for k, v in torch.load(model_weights_path).items() if "pretrained" in k},
        strict=False,
    )

    # Unfreeze all parameters and verify they require gradients
    for name, param in model.named_parameters():
        param.requires_grad = True

    custom_model_name = f"pretrained_l{lr}_e{epochs}_b{batch_size}_m{model_size}"

    # Set up TensorBoard writer
    writer = SummaryWriter(
        log_dir=os.path.join(
            logdir,
            custom_model_name,
        ),
    )

    # Set up optimizer and scheduler
    optim = torch.optim.AdamW(
        [
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "pretrained" in name
                ],
                "lr": lr,
            },
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "pretrained" not in name
                ],
                "lr": lr * 10,
            },
        ],
        lr=lr,
    )

    pre_params, params = 0, 0
    for name, param in model.named_parameters():
        if "pretrained" in name:
            pre_params += 1
        else:
            params += 1
    logger.info(
        "Pretrained params: %d, Non-pretrained params: %d",
        pre_params,
        params,
    )
    # exit()

    scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
        optimizer=optim,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=epochs * len(train_dataloader),
        # lr_end=1e-10,
    )

    # Create model savepoint directories
    state_path = os.path.join(
        os.path.join(output_path, custom_model_name),
        "cp",
    )
    save_model_path = os.path.join(
        os.path.join(output_path, custom_model_name),
        "model",
    )
    os.makedirs(state_path, exist_ok=True)
    os.makedirs(save_model_path, exist_ok=True)

    # Initialize EarlyStopping
    early_stopping = training_utils.EarlyStopping(
        patience=10,
        verbose=True,
        path=save_model_path,
    )

    model, optim, train_dataloader, val_dataloader, scheduler = accelerator.prepare(
        model,
        optim,
        train_dataloader,
        val_dataloader,
        scheduler,
    )

    criterion = nn.MSELoss()

    for epoch in range(epochs):

        train_loss, train_results = train(
            model=model,
            optim=optim,
            scheduler=scheduler,
            criterion=criterion,
            dataloader=train_dataloader,
            epoch=epoch,
            writer=writer,
            accelerator=accelerator,
            max_depth=max_depth,
        )

        val_loss, val_results = test(
            model=model,
            criterion=criterion,
            dataloader=val_dataloader,
            epoch=epoch,
            writer=writer,
            accelerator=accelerator,
            max_depth=max_depth,
        )

        # Log the current learning rate
        current_lr = optim.param_groups[0]["lr"]
        writer.add_scalar("Train/LearningRate", current_lr, epoch)

        accelerator.print(f"Epoch {epoch+1}")
        accelerator.print(
            f"Train Loss = {train_loss:.5f}, Train Metrics = {train_results}"
        )
        accelerator.print(f"Val Loss = {val_loss:.5f}, Val Metrics = {val_results}")

        accelerator.wait_for_everyone()
        accelerator.save_state(
            state_path,
            safe_serialization=False,
        )

        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            accelerator.print("Early stopping triggered")
            break

        accelerator.print("--------------------------------------------------")

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DepthAnythingV2 model on custom video dataset",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        # required=True,
        default="datasets/",
        help="Path to the directory containing input videos",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        # required=True,
        default="finetuning/",
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
        default=64,
        help="Batch size for training",
    )
    parser.add_argument(
        "-m",
        "--model_size",
        type=str,
        default="b",
        choices=["s", "b", "l"],
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
        "-ld",
        "--log_dir",
        type=str,
        required=True,
        help="Tensorboard logging directory",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)  # Ensure the output directory exists
    os.makedirs(args.log_dir, exist_ok=True)  # Ensure the logging directory exists

    main(
        args.input_dir,
        args.output_dir,
        args.batch_size,
        args.learning_rate,
        args.model_size,
        args.epochs,
        args.log_dir,
    )
