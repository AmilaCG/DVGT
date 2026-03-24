import argparse
import os
import time
import torch
import numpy as np
from dvgt.models.dvgt import DVGT
from iopath.common.file_io import g_pathmgr
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from PIL import Image
from torchvision import transforms as TF
from demo_viser import visualize_pred, launch_viser_server

from dvgt.utils.pose_enc import pose_encoding_to_ego_pose
from dvgt.utils.geometry import convert_point_in_ego_0_to_ray_depth_in_ego_n
from dvgt.utils.rotation import mat_to_quat
from scipy.spatial import cKDTree as KDTree
from nuscenes.utils.data_classes import LidarPointCloud

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

# =============================================================================
# Point Map Metrics
# Verbatim from https://github.com/yyfz/Pi3/blob/evaluation/mv_recon/utils.py
# =============================================================================

def accuracy(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(gt_points)
    distances, idx = gt_points_kd_tree.query(rec_points, workers=-1)
    acc = np.mean(distances)
    acc_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals[idx] * rec_normals, axis=-1)
        normal_dot = np.abs(normal_dot)
        return acc, acc_median, np.mean(normal_dot), np.median(normal_dot)

    return acc, acc_median


def completion(gt_points, rec_points, gt_normals=None, rec_normals=None):
    gt_points_kd_tree = KDTree(rec_points)
    distances, idx = gt_points_kd_tree.query(gt_points, workers=-1)
    comp = np.mean(distances)
    comp_median = np.median(distances)

    if gt_normals is not None and rec_normals is not None:
        normal_dot = np.sum(gt_normals * rec_normals[idx], axis=-1)
        normal_dot = np.abs(normal_dot)
        return comp, comp_median, np.mean(normal_dot), np.median(normal_dot)

    return comp, comp_median


# =============================================================================
# Ray Depth Metrics (DVGT paper Appendix B)
# =============================================================================

def ray_depth_absrel(d_gt, d_pred):
    """AbsRel on scalar ray depths: mean(|pred - gt| / gt) (↓ better)."""
    return float(np.mean(np.abs(d_pred - d_gt) / d_gt))


def ray_depth_delta(d_gt, d_pred, threshold=1.25):
    """Fraction of pixels with max(pred/gt, gt/pred) < threshold (↑ better)."""
    ratio = np.maximum(d_pred / d_gt, d_gt / d_pred)
    return float(np.mean(ratio < threshold))


# =============================================================================
# nuScenes LiDAR helpers
# =============================================================================

def get_lidar_points_in_ego_0(nusc, sample_tokens, world_to_ego_0):
    """
    Aggregate LIDAR_TOP points from all frames into ego_0 nuScenes frame.
    Returns np.ndarray of shape (N, 3).
    """
    all_points = []
    for token in sample_tokens:
        sample = nusc.get('sample', token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = nusc.get('sample_data', lidar_token)

        pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data['filename']))
        pts = pc.points[:3, :]  # (3, N)

        cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        R_l2e = Quaternion(cs['rotation']).rotation_matrix
        t_l2e = np.array(cs['translation'])
        pts_ego_n = R_l2e @ pts + t_l2e[:, None]  # (3, N) nuScenes ego frame

        ep = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_n_to_world = get_transform_matrix(ep['translation'], ep['rotation'])
        pts_h = np.vstack([pts_ego_n, np.ones((1, pts_ego_n.shape[1]))])  # (4, N)
        pts_ego_0 = (world_to_ego_0 @ (ego_n_to_world @ pts_h))[:3, :].T  # (N, 3)

        all_points.append(pts_ego_0)

    return np.vstack(all_points)


def get_sparse_ray_depth_gt_per_frame(nusc, sample_token, cameras, cam_img_sizes, target_size=512):
    """
    Project LIDAR_TOP points onto each camera to produce sparse GT ray depth maps.
    Ray depth = L2 norm of each point in the ego frame.

    Returns: list of (H_proc, W_proc) float32 arrays, one per camera in `cameras`.
             Zero entries indicate no LiDAR coverage at that pixel.
    """
    sample = nusc.get('sample', sample_token)
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)

    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data['filename']))
    pts_lidar = pc.points[:3, :]  # (3, N)

    cs_lidar = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    R_l2e = Quaternion(cs_lidar['rotation']).rotation_matrix
    t_l2e = np.array(cs_lidar['translation'])
    pts_ego = R_l2e @ pts_lidar + t_l2e[:, None]  # (3, N) nuScenes ego frame

    gt_ray_depth = np.linalg.norm(pts_ego, axis=0)  # (N,)

    sparse_maps = []
    for cam_idx, cam in enumerate(cameras):
        cam_data = nusc.get('sample_data', sample['data'][cam])
        cs_cam = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

        R_c2e = Quaternion(cs_cam['rotation']).rotation_matrix
        t_c2e = np.array(cs_cam['translation'])
        R_e2c = R_c2e.T
        t_e2c = -R_e2c @ t_c2e
        pts_cam = R_e2c @ pts_ego + t_e2c[:, None]  # (3, N)

        front = pts_cam[2, :] > 0
        pts_cam_f = pts_cam[:, front]
        d_f = gt_ray_depth[front]

        K = np.array(cs_cam['camera_intrinsic'])
        uv = K @ (pts_cam_f / pts_cam_f[2:3, :])  # (3, N) homogeneous
        u, v = uv[0, :], uv[1, :]

        orig_w, orig_h = cam_img_sizes[cam_idx]
        new_w = target_size
        new_h = round(orig_h * (new_w / orig_w) / 16) * 16
        u = u * (new_w / orig_w)
        v = v * (new_h / orig_h)

        if new_h > target_size:
            v = v - (new_h - target_size) // 2
            proc_h = target_size
        else:
            proc_h = new_h

        in_bounds = (u >= 0) & (u < new_w) & (v >= 0) & (v < proc_h)
        u_i = u[in_bounds].astype(np.int32)
        v_i = v[in_bounds].astype(np.int32)
        d_i = d_f[in_bounds]

        depth_map = np.zeros((proc_h, new_w), dtype=np.float32)
        # Sort descending so the nearest point overwrites farther ones at the same pixel
        order = np.argsort(d_i)[::-1]
        depth_map[v_i[order], u_i[order]] = d_i[order]
        sparse_maps.append(depth_map)

    return sparse_maps


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
            # print(f"Feeding image: {os.path.basename(img_path)}")
            frame_images.append(preprocess_image(img_path))
            with Image.open(img_path) as img:
                sizes.append(img.size)
        images.append(torch.stack(frame_images))
        cam_img_sizes.append(sizes)

    print(f"Feeding {len(images)} frames, each with {len(images[0])} views")
    images_tensor = torch.stack(images).unsqueeze(0).to(device) # With batch dimension
    # images_tensor = torch.stack(images).to(device) # Without batch dimension
    print(f"images_tensor: {images_tensor.shape}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        with torch.amp.autocast(device, dtype=dtype):
            # images (torch.Tensor): Input images with shape [T, V, 3, H, W] or [B, T, V, 3, H, W], in range [0, 1].
            # B: batch size, T: num_frames, V: views_per_frame, 3: RGB channels, H: height, W: width
            preds = model(images_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_time = time.perf_counter() - start_time

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

    # print(f"R errors (deg) — min: {r_errors.min():.3f}, max: {r_errors.max():.3f}, mean: {r_errors.mean():.3f}")
    # print(f"T errors (deg) — min: {t_errors.min():.3f}, max: {t_errors.max():.3f}, mean: {t_errors.mean():.3f}")
    auc30 = calculate_auc_np(r_errors, t_errors, max_threshold=30)
    print(f"Pose AUC@30: {auc30 * 100:.2f}")

    # Point Map Metrics (Accuracy & Completeness)
    print("\nComputing point map metrics...")
    gt_pts_ego0_nusc = get_lidar_points_in_ego_0(nusc, sample_tokens, world_to_ego_0)

    # world_points are in ego_0 OpenCV frame; rotate to nuScenes frame for comparison
    pred_pts_cv = preds['world_points'][0].cpu().float().numpy().reshape(-1, 3)
    R_cv_to_nusc = T_cv_to_nusc[:3, :3]
    pred_pts_nusc = pred_pts_cv @ R_cv_to_nusc.T

    acc, acc_median = accuracy(gt_pts_ego0_nusc, pred_pts_nusc)
    comp, comp_median = completion(gt_pts_ego0_nusc, pred_pts_nusc)
    print(f"Point Map Acc:  {acc:.4f} m  (median: {acc_median:.4f} m)")
    print(f"Point Map Comp: {comp:.4f} m  (median: {comp_median:.4f} m)")

    # Ray Depth Metrics (AbsRel & δ < 1.25)
    print("\nComputing ray depth metrics...")
    pred_ray_depth = convert_point_in_ego_0_to_ray_depth_in_ego_n(
        preds['world_points'], ego_n_to_ego_0
    )[0].cpu().numpy()  # (T, V, H, W)

    all_d_gt, all_d_pred = [], []
    for t_idx, token in enumerate(sample_tokens):
        sparse_maps = get_sparse_ray_depth_gt_per_frame(
            nusc, token, CAMERAS, cam_img_sizes[t_idx]
        )
        for v_idx, depth_map in enumerate(sparse_maps):
            mask = depth_map > 0
            if not mask.any():
                continue
            proc_h, proc_w = depth_map.shape
            pred_slice = pred_ray_depth[t_idx, v_idx, :proc_h, :proc_w]
            all_d_gt.append(depth_map[mask])
            all_d_pred.append(pred_slice[mask])

    if all_d_gt:
        d_gt = np.concatenate(all_d_gt)
        d_pred = np.concatenate(all_d_pred)
        print(f"Ray Depth AbsRel: {ray_depth_absrel(d_gt, d_pred):.4f}")
        print(f"Ray Depth δ<1.25: {ray_depth_delta(d_gt, d_pred):.4f}")

    print("\n" + "-" * 50)
    print("Profiling Metrics:")
    print(f"  Inference Time (for {args.frames}-frame seq): {inference_time:.4f} s")
    print(f"  Effective Inference FPS: {args.frames / inference_time:.2f}")
    if torch.cuda.is_available():
        peak_vram_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_vram_reserv = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        print(f"  Peak VRAM Allocated: {peak_vram_alloc:.2f} MB")
        print(f"  Peak VRAM Reserved:  {peak_vram_reserv:.2f} MB")

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