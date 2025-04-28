"Module for the evaluation metrics"

import logging
from typing import Optional
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch

# Setup logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_errors(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> dict:
    """
    Compute evaluation metrics for depth estimation.

    Args:
        pred (torch.Tensor): Predicted depth map
        gt (torch.Tensor): Ground truth depth map

    Returns:
        dict: Dictionary containing the computed metrics
    """
    assert pred.shape == gt.shape  # Shape 1D tensor

    # Add checks for NaN and Inf values
    if torch.isnan(pred).any():
        logger.warning("NaN values detected in predictions")
    if torch.isinf(pred).any():
        logger.warning("Inf values detected in predictions")

    # * 20 to get centimeters
    diff = pred - gt
    epsilon = 1e-6  # Small positive constant

    l1_error = torch.mean(torch.abs(diff))

    abs_rel = torch.mean(torch.abs(diff) / (gt + epsilon))  # Absolute relative error

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))  # Root mean squared error

    # δ<1.1 (percentage of pixels within 10% of actual depth)
    thresh = torch.max(
        (gt / pred),
        (pred / gt),
    )
    d1 = (thresh < 1.1).float().mean()

    return {
        "d1": d1.detach(),
        "abs_rel": abs_rel.detach(),
        "rmse": rmse.detach(),
        "l1": l1_error.detach(),
    }


def quaternion_distance(q1, q2):
    """
    Compute the geodesic distance between two unit quaternions,
    returning the angular error in degrees.

    Args:
        q1 (np.ndarray): First quaternion, shape (4,).
        q2 (np.ndarray): Second quaternion, shape (4,).

    Returns:
        float: Angle (in degrees) between the two rotations.
    """
    # Ensure quaternions are normalized
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    # Dot product might be negative; take absolute value
    dot_val = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    angle_rad = 2 * np.arccos(dot_val)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def compute_ate(gt_trans, pred_trans):
    """
    Compute Absolute Trajectory Error (ATE) for translation.

    Args:
        gt_trans (np.ndarray): Ground truth positions, shape (N, 3).
        pred_trans (np.ndarray): Predicted positions, shape (N, 3).

    Returns:
        float: ATE (root mean square error) in the same units as gt_trans.
    """
    errors = np.linalg.norm(gt_trans - pred_trans, axis=1)
    ate = np.sqrt(np.mean(errors**2))
    return ate


def compute_rte(gt_trans, pred_trans):
    """
    Compute Relative Translation Error (RTE) as the average error
    between relative translations computed on consecutive frames.

    Args:
        gt_trans (np.ndarray): Ground truth positions, shape (N, 3).
        pred_trans (np.ndarray): Predicted positions, shape (N, 3).

    Returns:
        float: RTE (average error between relative translations).
    """
    # Compute relative translations between consecutive frames
    gt_rel = np.diff(gt_trans, axis=0)
    pred_rel = np.diff(pred_trans, axis=0)

    errors = np.linalg.norm(gt_rel - pred_rel, axis=1)
    rte = np.mean(errors)
    return rte


def compute_rot_error(gt_rots, pred_rots):
    """
    Compute rotation error as per paper definition:
    ROT = μτ((trace(Rot(Ωτ^-1Ω'τ)) - 1) * 180/π/2)
    """
    errors = []
    for q_gt, q_pred in zip(gt_rots, pred_rots):
        try:
            # Check if predicted quaternion is zero
            q_pred_norm = np.linalg.norm(q_pred)
            if q_pred_norm < 1e-8:
                # If zero, use identity quaternion [0, 0, 0, 1]
                q_pred = np.array([0.0, 0.0, 0.0, 1.0])
                logger.warning(
                    f"Zero quaternion detected, using identity quaternion instead"
                )

            # Normalize quaternions for numerical stability
            q_gt = q_gt / np.linalg.norm(q_gt)
            q_pred = q_pred / np.linalg.norm(q_pred)

            R_gt = R.from_quat(q_gt).as_matrix()
            R_pred = R.from_quat(q_pred).as_matrix()

            # Compute relative rotation
            R_rel = R_gt.T @ R_pred

            # Calculate angle in degrees
            cos_theta = (np.trace(R_rel) - 1) / 2
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta_deg = np.degrees(np.arccos(cos_theta))
            errors.append(theta_deg)

        except ValueError as e:
            logger.warning(f"Error computing rotation: {e}")
            logger.warning(f"q_gt: {q_gt}, norm: {np.linalg.norm(q_gt)}")
            logger.warning(f"q_pred: {q_pred}, norm: {np.linalg.norm(q_pred)}")
            errors.append(180.0)  # Maximum possible rotation error

    return np.mean(errors)


def compute_pose_errors(
    pred_positions: torch.Tensor,
    gt_positions: torch.Tensor,
) -> dict:
    """
    Compute pose errors between predicted and ground truth positions.

    Args:
        pred_positions (torch.Tensor): Predicted positions (batch_size, 7)
        gt_positions (torch.Tensor): Ground truth positions (batch_size, 7)

    Returns:
        dict: Dictionary containing ATE, RTE, and ROTE metrics
    """
    # Convert to numpy and ensure contiguous memory layout
    pred_t = pred_positions[:, :3].cpu().numpy().copy()
    pred_q = pred_positions[:, 3:].cpu().numpy().copy()
    gt_t = gt_positions[:, :3].cpu().numpy().copy()
    gt_q = gt_positions[:, 3:].cpu().numpy().copy()

    # Normalize quaternions before computing errors
    pred_q_norms = np.linalg.norm(pred_q, axis=1, keepdims=True)
    gt_q_norms = np.linalg.norm(gt_q, axis=1, keepdims=True)

    # Use small epsilon for numerical stability
    epsilon = 1e-8
    pred_q_norms = np.maximum(pred_q_norms, epsilon)
    gt_q_norms = np.maximum(gt_q_norms, epsilon)

    pred_q = pred_q / pred_q_norms
    gt_q = gt_q / gt_q_norms

    # Ensure quaternions are in the same hemisphere (prevent double coverage)
    dot_products = np.sum(gt_q * pred_q, axis=1)
    pred_q[dot_products < 0] *= -1  # Flip quaternions if needed

    ate = compute_ate(gt_t, pred_t)
    rte = compute_rte(gt_t, pred_t)
    rote = compute_rot_error(gt_q, pred_q)

    return {
        "ate": torch.as_tensor(ate),
        "rte": torch.as_tensor(rte),
        "rote": torch.as_tensor(rote),
    }


def evaluate_trajectory(
    pred_rel_poses: torch.Tensor,
    gt_rel_poses: torch.Tensor,
    initial_pose: Optional[torch.Tensor] = None,
):
    """
    Evaluate the predicted trajectory using both relative and absolute metrics.

    Args:
        pred_rel_poses: Predicted relative poses (N, 7). Here N is the number of frames.
        gt_rel_poses: Ground truth relative poses (N, 7). Here N is the number of frames.
        initial_pose: Optional initial pose

    Returns:
        dict containing ATE, RTE, and ROT metrics
    """
    # Calculate scale factor
    scale = calculate_scale_factor(pred_rel_poses, gt_rel_poses)

    # Scale the predicted relative poses
    scaled_pred_rel = pred_rel_poses.clone()  # (N, 7)
    scaled_pred_rel[:, :3] *= scale  # (N, 3)

    # Compose poses to get absolute trajectories
    pred_abs_poses = compose_poses(scaled_pred_rel, initial_pose)  # (N+1, 7)
    gt_abs_poses = compose_poses(gt_rel_poses, initial_pose)  # (N+1, 7)

    # Calculate metrics
    metrics = {
        "rte": compute_rte(
            scaled_pred_rel[:, :3],
            gt_rel_poses[:, :3],
        ),
        "ate": compute_ate(
            gt_abs_poses[:, :3],
            pred_abs_poses[:, :3],
        ),
        "rote": compute_rot_error(
            gt_abs_poses[:, 3:],
            pred_abs_poses[:, 3:],
        ),
    }

    return metrics


def calculate_scale_factor(pred_rel_poses, gt_rel_poses):
    """
    Calculate scale factor according to equation (6) in the paper.

    Args:
        pred_rel_poses: Predicted relative poses (N, 7)
        gt_rel_poses: Ground truth relative poses (N, 7)

    Returns:
        Scale factor (scalar)
    """
    pred_trans = pred_rel_poses[:, :3]  # Get translations only
    gt_trans = gt_rel_poses[:, :3]

    # Calculate scale factor
    numerator = torch.sum(pred_trans * gt_trans)
    denominator = torch.sum(pred_trans * pred_trans)

    scale = numerator / denominator
    return scale


def compose_poses(
    relative_poses: torch.Tensor,
    initial_pose: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compose relative poses to get absolute poses.
    P_τ = P₁Ω₁...Ωτ₋₁

    Args:
        relative_poses: Tensor of shape (N, 7) or (batch_size, N, 7) containing relative poses [tx, ty, tz, qx, qy, qz, qw]
        initial_pose: Optional initial pose P₁. If None, identity pose is used.

    Returns:
        Tensor of shape (N+1, 7) containing absolute poses
    """
    device = relative_poses.device
    if initial_pose is None:
        # Identity pose: [0,0,0] translation and [0,0,0,1] quaternion
        initial_pose = torch.tensor(
            [0, 0, 0, 0, 0, 0, 1], device=device, dtype=torch.float32
        )

    # Ensure initial_pose is a 1D tensor
    if initial_pose.dim() > 1:
        initial_pose = initial_pose.squeeze()

    # Handle 3D input tensor (batch_size, N, 7)
    if relative_poses.dim() == 3:
        # For now, just use the first batch
        relative_poses = relative_poses[0]  # Shape: (N, 7)
        # print(f"Reshaped relative_poses from 3D to 2D: {relative_poses.shape}")

    # Ensure relative_poses is a 2D tensor
    if relative_poses.dim() == 1:
        relative_poses = relative_poses.unsqueeze(0)

    # Print shapes for debugging
    # print(f"initial_pose shape: {initial_pose.shape}")
    # print(f"relative_poses shape: {relative_poses.shape}")

    absolute_poses = [initial_pose]
    current_pose = initial_pose

    for rel_pose in relative_poses:
        # Extract translation and rotation from current pose
        curr_trans = current_pose[:3]  # (3,)
        curr_quat = current_pose[3:]  # (4,)

        # Extract translation and rotation from relative pose
        rel_trans = rel_pose[:3]  # (3,)
        rel_quat = rel_pose[3:]  # (4,)

        # Check if relative quaternion is zero and replace with identity if needed
        rel_quat_norm = torch.norm(rel_quat)
        if rel_quat_norm < 1e-8:
            # Replace with identity quaternion [0, 0, 0, 1]
            rel_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
            logger.warning(
                f"Zero quaternion detected in relative pose, using identity quaternion instead"
            )

        # Compose rotations (quaternion multiplication)
        new_quat = quaternion_multiply(curr_quat, rel_quat)  # (4,)

        # Compose translations: new_t = curr_t + curr_R * rel_t
        # Ensure dimensions match for quaternion_rotate_vector
        rel_trans_rotated = quaternion_rotate_vector(curr_quat, rel_trans)  # (3,)
        new_trans = curr_trans + rel_trans_rotated  # (3,)

        # Ensure new_trans is a vector of size 3 and new_quat is a vector of size 4
        if new_trans.numel() != 3:
            new_trans = new_trans[:3]
        if new_quat.numel() != 4:
            new_quat = new_quat[:4]

        # print(f"new_trans shape before applying the transformation: {new_trans.shape}")
        # print(f"new_quat shape before applying the transformation: {new_quat.shape}")

        # Ensure both tensors have the correct shape before concatenation
        # If they're 2D, take the first row or column depending on shape
        if new_trans.dim() > 1:
            if new_trans.shape[0] == 3:
                new_trans = new_trans[0]  # Take first row if shape is (3, N)
            else:
                new_trans = new_trans[:, 0]  # Take first column if shape is (N, 3)

        if new_quat.dim() > 1:
            if new_quat.shape[0] == 4:
                new_quat = new_quat[0]  # Take first row if shape is (4, N)
            else:
                new_quat = new_quat[:, 0]  # Take first column if shape is (N, 4)

        # print(f"new_trans shape before applying the transformation: {new_trans.shape}")
        # print(f"new_quat shape before applying the transformation: {new_quat.shape}")

        # Combine into new pose - both should be 1D tensors
        new_pose = torch.cat([new_trans, new_quat])

        # print(f"new_pose shape: {new_pose.shape}")

        absolute_poses.append(new_pose)
        current_pose = new_pose

    return torch.stack(absolute_poses)


def quaternion_multiply(
    q1: torch.Tensor,
    q2: torch.Tensor,
) -> torch.Tensor:
    """
    Multiply two quaternions.

    Args:
        q1: First quaternion [qx, qy, qz, qw]
        q2: Second quaternion [qx, qy, qz, qw]

    Returns:
        Product quaternion [qx, qy, qz, qw]
    """
    # Store original dimensions to determine output shape
    q1_orig_dim = q1.dim()
    q2_orig_dim = q2.dim()

    # Handle different dimensions
    if q1.dim() != q2.dim():
        # If q1 is a 1D tensor and q2 is a 2D tensor, expand q1
        if q1.dim() == 1 and q2.dim() == 2:
            q1 = q1.unsqueeze(0)
        # If q2 is a 1D tensor and q1 is a 2D tensor, expand q2
        elif q2.dim() == 1 and q1.dim() == 2:
            q2 = q2.unsqueeze(0)

    w1, x1, y1, z1 = q1[..., 3], q1[..., 0], q1[..., 1], q1[..., 2]
    w2, x2, y2, z2 = q2[..., 3], q2[..., 0], q2[..., 1], q2[..., 2]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Handle different output dimensions based on input
    if q1_orig_dim == 1 and q2_orig_dim == 1:
        return torch.stack([x, y, z, w])
    else:
        return torch.stack([x, y, z, w], dim=-1)


def quaternion_rotate_vector(
    quat: torch.Tensor,
    vec: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate a vector by a quaternion.

    Args:
        quat: Quaternion [qx, qy, qz, qw] of shape (4,) or (batch_size, 4)
        vec: Vector to rotate [x, y, z] of shape (3,) or (batch_size, 3)

    Returns:
        Rotated vector of shape (3,) or (batch_size, 3)
    """
    # Store original dimensions to determine output shape
    quat_orig_dim = quat.dim()
    vec_orig_dim = vec.dim()

    # Extract the vector part of the quaternion
    qvec = quat[:3]  # Shape: (3,) or (batch_size, 3)
    qw = quat[3]  # Shape: () or (batch_size,)

    # If vec has a size of 7 at dimension 1, extract only the first 3 elements
    if isinstance(vec, torch.Tensor) and vec.shape[-1] == 7:
        vec = vec[..., :3]  # Shape: (3,) or (batch_size, 3)

    # Handle different dimensions between qvec and vec
    if qvec.dim() != vec.dim():
        # If qvec is 1D and vec is 2D, expand qvec to match vec's batch dimension
        if qvec.dim() == 1 and vec.dim() == 2:
            qvec = qvec.unsqueeze(0).expand(vec.shape[0], -1)  # Shape: (batch_size, 3)
            qw = qw.unsqueeze(0).expand(vec.shape[0])  # Shape: (batch_size,)
        # If vec is 1D and qvec is 2D, expand vec to match qvec's batch dimension
        elif vec.dim() == 1 and qvec.dim() == 2:
            vec = vec.unsqueeze(0).expand(qvec.shape[0], -1)  # Shape: (batch_size, 3)

    # Perform the rotation
    uv = torch.cross(qvec, vec)  # Shape: (3,) or (batch_size, 3)
    uuv = torch.cross(qvec, uv)  # Shape: (3,) or (batch_size, 3)

    # Handle broadcasting for qw
    if qw.dim() == 0 and vec.dim() == 1:
        # If qw is a scalar and vec is 1D, expand qw to match vec's shape
        qw = qw.unsqueeze(0).expand(3)  # Shape: (3,)
    elif qw.dim() == 0 and vec.dim() == 2:
        # If qw is a scalar and vec is 2D, expand qw to match vec's batch dimension
        qw = qw.unsqueeze(0).expand(vec.shape[0])  # Shape: (batch_size,)

    # Ensure qw has the right shape for broadcasting
    if qw.dim() == 1 and vec.dim() == 2:
        qw = qw.unsqueeze(1)  # Shape: (batch_size, 1)

    result = vec + 2 * (uv * qw + uuv)  # Shape: (3,) or (batch_size, 3)

    # If both inputs were 1D, ensure output is 1D
    if quat_orig_dim == 1 and vec_orig_dim == 1:
        result = result.squeeze()

    return result
