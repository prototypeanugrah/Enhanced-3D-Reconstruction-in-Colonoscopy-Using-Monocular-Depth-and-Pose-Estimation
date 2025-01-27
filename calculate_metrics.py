import os
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import glob


def load_depth_map(file_path):
    """Load depth map from file."""
    depth = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
    if depth is None:
        raise ValueError(f"Could not load depth map: {file_path}")
    return depth.astype(np.float32) / 1000.0  # Convert mm to meters


def calculate_metrics(gt, pred, mask_invalid=True):
    """Calculate depth estimation metrics."""
    if mask_invalid:
        valid_mask = (gt > 0) & (pred > 0) & (~np.isinf(gt)) & (~np.isinf(pred))
        gt = gt[valid_mask]
        pred = pred[valid_mask]

    # Handle empty or invalid inputs
    if len(gt) == 0 or len(pred) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "abs_rel": np.nan,
            "sq_rel": np.nan,
            "delta1": np.nan,
            "delta2": np.nan,
            "delta3": np.nan,
        }

    # Calculate metrics
    thresh = np.maximum((gt / pred), (pred / gt))
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25**2).mean()
    delta3 = (thresh < 1.25**3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    mae = np.abs(gt - pred).mean()
    abs_rel = np.abs(gt - pred).mean() / gt.mean()
    sq_rel = ((gt - pred) ** 2).mean() / gt.mean()

    return {
        "rmse": rmse,
        "mae": mae,
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }


def process_scene(gt_folder, pred_folder):
    """Process a single scene and calculate metrics."""
    metrics_list = []

    # Get all ground truth files
    gt_files = sorted(Path(gt_folder).glob("*.png"))

    for gt_file in tqdm(gt_files, desc=f"Processing {Path(gt_folder).name}"):
        # Construct predicted depth file path
        pred_file = Path(pred_folder) / gt_file.name

        if not pred_file.exists():
            print(f"Warning: Missing prediction for {gt_file.name}")
            continue

        # Load depth maps
        gt_depth = load_depth_map(str(gt_file))
        pred_depth = load_depth_map(str(pred_file))

        # Calculate metrics for this frame
        frame_metrics = calculate_metrics(gt_depth, pred_depth)
        metrics_list.append(frame_metrics)

    # Calculate average metrics for the scene
    if metrics_list:
        avg_metrics = {
            metric: np.mean([m[metric] for m in metrics_list])
            for metric in metrics_list[0].keys()
        }
        return avg_metrics
    return None


def main():
    # Base directory
    dataset_root = "./datasets/SyntheticColon/"

    # Get all SyntheticColon_* directories
    colon_dirs = sorted(glob.glob(os.path.join(dataset_root, "SyntheticColon_*")))

    all_results = {}

    for colon_dir in colon_dirs:
        colon_name = os.path.basename(colon_dir)
        print(f"\nProcessing {colon_name}")

        # Get all procedure directories (excluding *_OP directories)
        procedure_dirs = [
            d
            for d in glob.glob(os.path.join(colon_dir, "Frames_*"))
            if not d.endswith("_OP")
        ]

        results = {}

        for proc_dir in procedure_dirs:
            proc_name = os.path.basename(proc_dir)
            pred_dir = proc_dir + "_OP"

            if not os.path.exists(pred_dir):
                print(f"Warning: Missing predictions directory for {proc_name}")
                continue

            print(f"\nProcessing procedure {proc_name}")
            scene_metrics = process_scene(proc_dir, pred_dir)

            if scene_metrics:
                results[proc_name] = scene_metrics

        all_results[colon_name] = results

        # Print results for this SyntheticColon directory
        print(f"\nResults for {colon_name}:")
        print("-" * 50)
        for proc, metrics in results.items():
            print(f"\nProcedure {proc}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")

        # Calculate and print average for this SyntheticColon directory
        if results:
            print(f"\nAverage for {colon_name}:")
            print("-" * 50)
            avg_metrics = {
                metric: np.mean([results[proc][metric] for proc in results.keys()])
                for metric in results[list(results.keys())[0]].keys()
            }
            for metric_name, value in avg_metrics.items():
                print(f"{metric_name}: {value:.4f}")

    # Calculate and print overall average across all SyntheticColon directories
    print("\nOverall Average across all SyntheticColon directories:")
    print("-" * 50)

    # Flatten all metrics
    all_metrics = []
    for colon_results in all_results.values():
        all_metrics.extend(list(colon_results.values()))

    if all_metrics:
        overall_avg = {
            metric: np.mean([m[metric] for m in all_metrics])
            for metric in all_metrics[0].keys()
        }
        for metric_name, value in overall_avg.items():
            print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()
