import os
import sys

from PIL import Image
import numpy as np
import glob

INPUT_PATH = "./datasets/"

warning1 = False
warning2 = False


def check_depth(pred):
    global warning1, warning2
    assert pred.shape == (
        475,
        475,
    ), "Wrong size of predicted depth, expected [475,475], got {}".format(
        list(pred.shape)
    )
    assert pred.dtype == np.float16, "Wrong data type, expected float16, got {}".format(
        pred.dtype
    )
    if np.max(pred) > 1:
        if not warning1:
            print("Warning: Depths > 20cm encountered")
        warning1 = True
    if np.min(pred) < 0:
        if not warning2:
            print("Warning: Depths < 0cm encountered")
        warning2 = True

    return pred.clip(0, 1)  # depths are clipped to (0,1) to avoid invalid depths


def load_depth(pred_file, gt_file):
    gt_depth = (
        np.array(Image.open(gt_file.replace("FrameBuffer", "Depth"))) / 255 / 256
    )  # please use this to load ground truth depth during training and testing
    pred = np.load(pred_file)
    pred = check_depth(pred)

    # Resize ground truth to match prediction size
    gt_depth = np.array(
        Image.fromarray(gt_depth).resize(
            (pred.shape[1], pred.shape[0]),
            Image.BILINEAR,
        )
    )
    return pred, gt_depth


def eval_depth(pred, gt_depth):
    # * 20 to get centimeters
    diff = pred - gt_depth
    epsilon = 1e-6  # Small positive constant
    L1_error = np.mean(np.abs(diff))
    abs_rel = np.mean(np.abs(diff) / (gt_depth + epsilon))
    RMSE_error = np.sqrt(np.mean((diff) ** 2))
    # δ<1.1 (percentage of pixels within 10% of actual depth)
    thresh = np.maximum(
        (gt_depth / pred),
        (pred / gt_depth),
    )
    d1 = np.mean(thresh < 1.1)
    return L1_error, abs_rel, d1, RMSE_error


def process_depths(test_folders, INPUT_PATH):
    # first check if all the data is there
    for traj in test_folders:
        # print(traj)
        assert os.path.exists(INPUT_PATH + traj + "/depth/"), "No input folder found"
        input_file_list = np.sort(
            glob.glob(INPUT_PATH + traj + "/depth/FrameBuffer*.npy")
        )
        if traj[18] == "I":
            assert len(input_file_list) == 601, "Predictions missing in {}".format(traj)
        else:
            assert len(input_file_list) == 1201, "Predictions missing in {}".format(
                traj
            )

    # loop through predictions
    for traj in test_folders:
        print("Processing ", traj)
        input_file_list = np.sort(
            glob.glob(INPUT_PATH + traj + "/depth/FrameBuffer*.npy")
        )
        L1_errors, abs_rels, d1_err, rmses = [], [], [], []
        preds, gts = [], []
        for i in range(len(input_file_list)):
            file_name1 = input_file_list[i].split("/")[-1]
            # print(file_name1)
            im1_path = input_file_list[i]
            gt_depth_path = (
                INPUT_PATH + traj.strip("_OP") + "/" + file_name1.replace("npy", "png")
            )
            pred_depth, gt_depth = load_depth(im1_path, gt_depth_path)
            preds.append(pred_depth)
            gts.append(gt_depth)
        gts_ = np.mean(np.mean(np.array(gts), 1), 1)
        preds_ = np.mean(np.mean(np.array(preds), 1), 1)
        scale = np.sum(preds_ * gts_) / np.sum(
            preds_ * preds_
        )  # monocular methods predict depth up to scale
        print("Scale: ", scale)

        for i in range(len(input_file_list)):
            L1_error, abs_rel, d1, rmse = eval_depth(
                preds[i] * scale,
                gts[i],
            )
            L1_errors.append(L1_error)
            abs_rels.append(abs_rel)
            d1_err.append(d1)
            rmses.append(rmse)
        print("Mean L1 error in cm: ", np.mean(L1_errors))
        print("Mean AbsRel error in cm: ", np.mean(abs_rels))
        print("Median d1 error in cm: ", np.mean(d1_err))
        print("Mean RMSE in cm: ", np.mean(rmses))


def main():
    # The 9 test sequences have to organized in the submission .zip file as follows:
    test_folders = [
        "/SyntheticColon_I/Frames_S5_OP",
        "/SyntheticColon_I/Frames_S10_OP",
        "/SyntheticColon_I/Frames_S15_OP",
        "/SyntheticColon_II/Frames_B5_OP",
        "/SyntheticColon_II/Frames_B10_OP",
        "/SyntheticColon_II/Frames_B15_OP",
        "/SyntheticColon_III/Frames_O1_OP",
        "/SyntheticColon_III/Frames_O2_OP",
        "/SyntheticColon_III/Frames_O3_OP",
    ]

    process_depths(test_folders, INPUT_PATH)


if __name__ == "__main__":
    main()
