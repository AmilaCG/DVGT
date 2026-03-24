import argparse
import os
import torch
import numpy as np
from dvgt.models.dvgt import DVGT
from dvgt.utils.load_fn import load_and_preprocess_images
from iopath.common.file_io import g_pathmgr
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from PIL import Image
from torchvision import transforms as TF
from demo_viser import visualize_pred, launch_viser_server

from dvgt.utils.pose_enc import pose_encoding_to_ego_pose
from dvgt.utils.geometry import convert_point_in_ego_0_to_ray_depth_in_ego_n
from dvgt.utils.rotation import mat_to_quat

# Predict in alphabetical order to match standard loading
CAMERAS = [
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
]

# =============================================================================
# Pose Evaluation (adapted from Depth-Anything-3/src/depth_anything_3/bench/utils.py)
# =============================================================================

def build_pair_index(N: int, B: int = 1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = ((i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_])
    return i1, i2


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    Uses closed-form solution instead of torch.inverse() for numerical stability.

    Args:
        se3: Nx4x4 or Nx3x4 tensor of SE3 matrices
        R: Optional Nx3x3 rotation matrices
        T: Optional Nx3x1 translation vectors

    Returns:
        Inverted SE3 matrices with same shape as input
    """
    is_numpy = isinstance(se3, np.ndarray)

    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]

    if is_numpy:
        R_transposed = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)
        top_right = -torch.bmm(R_transposed, T)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=False):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity (disabled for ego-motion)

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.

    Args:
        pred_se3: Predicted SE(3) transformations
        gt_se3: Ground truth SE(3) transformations
        num_frames: Number of frames

    Returns:
        Tuple of (rotation angle errors, translation angle errors) in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)

    # Compute relative camera poses between pairs using closed-form inverse
    relative_pose_gt = closed_form_inverse_se3(gt_se3[pair_idx_i1]).bmm(gt_se3[pair_idx_i2])
    relative_pose_pred = closed_form_inverse_se3(pred_se3[pair_idx_i1]).bmm(pred_se3[pair_idx_i2])

    # Compute the difference in rotation and translation
    rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
    rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])

    return rel_rangle_deg, rel_tangle_deg


def get_transform_matrix(translation, rotation):
    """Convert translation vector and rotation quaternion to a 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T

# Took from https://github.com/facebookresearch/PoseDiffusion/blob/main/pose_diffusion/util/metric.py
def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays.

    :param r_error: numpy array representing R error values (Degree).
    :param t_error: numpy array representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error arrays along a new axis
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram))

def get_transform_matrix(translation, rotation):
    """Convert translation vector and rotation quaternion to a 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T

def preprocess_image(img_path, target_size=512):
    """Preprocess image matching DVGT's aspect ratio and cropping logic."""
    img = Image.open(img_path).convert("RGB")
    width, height = img.size
    new_width = target_size
    new_height = round(height * (new_width / width) / 16) * 16
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    
    to_tensor = TF.ToTensor()
    img_t = to_tensor(img)
    
    if new_height > target_size:
        start_y = (new_height - target_size) // 2
        img_t = img_t[:, start_y : start_y + target_size, :]
        
    return img_t

def main(args):
    checkpoint_path = 'ckpt/open_ckpt.pt'

    device = "cuda"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Initialize the model and load the pretrained weights.
    model = DVGT()
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    # Load and preprocess example images (replace with your own image paths)
    # image_dir = 'examples/openscene_log-0104-scene-0007'
    # images = load_and_preprocess_images(image_dir, start_frame=16, end_frame=23).to(device)

    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=True)
    scene = nusc.scene[4]
    sample_token = scene['first_sample_token']
    sample_tokens = []
    while sample_token != '':
        sample_tokens.append(sample_token)
        sample = nusc.get('sample', sample_token)
        sample_token = sample['next']

    print(f"Sample tokens: {len(sample_tokens)}")
    # Cap sample tokens
    sample_tokens = sample_tokens[:args.frames]
    T = len(sample_tokens)
    
    images, cam_img_sizes = [], []
    for token in sample_tokens:
        sample = nusc.get('sample', token)
        frame_images, sizes = [], []
        for cam in CAMERAS:
            cam_data = nusc.get('sample_data', sample['data'][cam])
            img_path = os.path.join(nusc.dataroot, cam_data['filename'])
            print(f"Feeding image: {os.path.basename(img_path)}")
            frame_images.append(preprocess_image(img_path))
            with Image.open(img_path) as img:
                sizes.append(img.size)
        images.append(torch.stack(frame_images))
        cam_img_sizes.append(sizes)

    print(f"Feeding {len(images)} frames, each with {len(images[0])} views")
    images_tensor = torch.stack(images).unsqueeze(0).to(device) # With batch dimension
    # images_tensor = torch.stack(images).to(device) # Without batch dimension
    print(f"images_tensor: {images_tensor.shape}")

    with torch.no_grad():
        with torch.amp.autocast(device, dtype=dtype):
            # images (torch.Tensor): Input images with shape [T, V, 3, H, W] or [B, T, V, 3, H, W], in range [0, 1].
            # B: batch size, T: num_frames, V: views_per_frame, 3: RGB channels, H: height, W: width
            preds = model(images_tensor)

    ego_n_to_ego_0 = pose_encoding_to_ego_pose(preds['ego_pose_enc'])
    pred_ego_poses = ego_n_to_ego_0[0].cpu().numpy()  # (T, 3, 4)

    pred_poses_44 = np.zeros((T, 4, 4))
    pred_poses_44[:, :3, :] = pred_ego_poses
    pred_poses_44[:, 3, 3] = 1.0

    # Compute GT poses (ego_n_to_ego_0) from nuScenes
    sample_0 = nusc.get('sample', sample_tokens[0])
    ego_pose_0 = nusc.get('ego_pose', nusc.get('sample_data', sample_0['data']['CAM_FRONT'])['ego_pose_token'])
    world_to_ego_0 = np.linalg.inv(get_transform_matrix(ego_pose_0['translation'], ego_pose_0['rotation']))

    gt_poses_44 = []
    for token in sample_tokens:
        sample = nusc.get('sample', token)
        ego_pose_n = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['CAM_FRONT'])['ego_pose_token'])
        ego_n_to_world = get_transform_matrix(ego_pose_n['translation'], ego_pose_n['rotation'])
        gt_poses_44.append(world_to_ego_0 @ ego_n_to_world)
    gt_poses_44 = np.stack(gt_poses_44)

    # Convert predicted poses from OpenCV to nuScenes convention
    R_nusc_to_cv = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
    T_nusc_to_cv = np.eye(4, dtype=np.float64)
    T_nusc_to_cv[:3, :3] = R_nusc_to_cv
    T_cv_to_nusc = np.linalg.inv(T_nusc_to_cv)
    pred_poses_nusc = np.array([T_cv_to_nusc @ p @ T_nusc_to_cv for p in pred_poses_44])

    # Compute pairwise rotation and translation angle errors
    pred_t = torch.from_numpy(pred_poses_nusc).float()
    gt_t = torch.from_numpy(gt_poses_44).float()
    r_errors, t_errors = se3_to_relative_pose_error(pred_t, gt_t, T)
    r_errors = r_errors.numpy()
    t_errors = t_errors.numpy()

    print(f"R errors (deg) — min: {r_errors.min():.3f}, max: {r_errors.max():.3f}, mean: {r_errors.mean():.3f}")
    print(f"T errors (deg) — min: {t_errors.min():.3f}, max: {t_errors.max():.3f}, mean: {t_errors.mean():.3f}")
    auc30 = calculate_auc_np(r_errors, t_errors, max_threshold=30)
    print(f"Pose AUC@30: {auc30 * 100:.2f}")

    if (args.vis):
        vis_args = argparse.Namespace(
            conf_threshold=25.0,
            mask_sky=False,
            use_edge_masks=False,
            edge_depth_rtol=0.1,
            edge_normal_tol=50,
            max_depth=-1,
            downsample_ratio=-1,
        )
        point_clouds, poses = visualize_pred(preds, vis_args)
        launch_viser_server(point_clouds, poses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default=os.path.expanduser('~/datasets/nuscenes-mini'))
    # parser.add_argument('--checkpoint', type=str, default='ckpt/open_ckpt.pt')
    parser.add_argument('--frames', type=int, default=16, choices=range(1, 25), help="Frames to evaluate per scene (Max 24).")
    # parser.add_argument('--frames_chunk_size', type=int, default=8)
    parser.add_argument('--vis', action='store_true', help='Visualize using Viser')
    args = parser.parse_args()
    main(args)