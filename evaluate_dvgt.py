import argparse
import os
import time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms as TF

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points

from dvgt.models.dvgt import DVGT
from dvgt.utils.pose_enc import pose_encoding_to_ego_pose
from dvgt.utils.geometry import convert_point_in_ego_0_to_ray_depth_in_ego_n

# Predict in alphabetical order to match standard loading
CAMERAS = [
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
]

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

def get_transform_matrix(translation, rotation):
    """Convert translation vector and rotation quaternion to a 4x4 transform."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T

def compute_pose_auc_30(pred_poses, gt_poses):
    """Computes AUC@30 for pose following Appendix B."""
    T = pred_poses.shape[0]
    if T < 2:
        return 0.0

    # Transform predicted poses (OpenCV convention) to nuScenes convention
    R_nusc_to_cv = np.array([
        [ 0, -1,  0],
        [ 0,  0, -1],
        [ 1,  0,  0]
    ], dtype=np.float64)
    T_nusc_to_cv = np.eye(4, dtype=np.float64)
    T_nusc_to_cv[:3, :3] = R_nusc_to_cv
    T_cv_to_nusc = np.linalg.inv(T_nusc_to_cv)
    
    pred_poses_nusc = np.array([T_cv_to_nusc @ p @ T_nusc_to_cv for p in pred_poses])

    errors = []
    for i in range(T):
        for j in range(T):
            if i == j: continue
            
            pred_rel = np.linalg.inv(pred_poses_nusc[i]) @ pred_poses_nusc[j]
            gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[j]

            R_err = pred_rel[:3, :3] @ np.linalg.inv(gt_rel[:3, :3])
            trace = np.clip(np.trace(R_err), -1.0, 3.0)
            rra = np.degrees(np.arccos((trace - 1.0) / 2.0))

            t_pred = pred_rel[:3, 3]
            t_gt = gt_rel[:3, 3]
            
            n_pred = np.linalg.norm(t_pred)
            n_gt = np.linalg.norm(t_gt)
            
            if n_pred > 1e-6 and n_gt > 1e-6:
                cos_theta = np.clip(np.dot(t_pred, t_gt) / (n_pred * n_gt), -1.0, 1.0)
                rta = np.degrees(np.arccos(cos_theta))
            else:
                rta = 0.0

            errors.append(max(rra, rta))

    if not errors: return 0.0
    errors = np.array(errors)

    thresholds = np.linspace(0, 30, 100)
    acc_list = [np.mean(errors < th) for th in thresholds]
    return np.trapz(acc_list, thresholds) / 30.0 * 100.0

def project_ray_depth_to_image(nusc, lidar_token, cam_token):
    cam_data = nusc.get('sample_data', cam_token)
    lidar_data = nusc.get('sample_data', lidar_token)
    
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data['filename']))
    
    lidar_cs = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pc.rotate(Quaternion(lidar_cs['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_cs['translation']))
    
    lidar_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    pc.rotate(Quaternion(lidar_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_pose['translation']))
    
    cam_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
    pc.translate(-np.array(cam_pose['translation']))
    pc.rotate(Quaternion(cam_pose['rotation']).rotation_matrix.T)
    
    # Ray Depth: distance to the camera's ego-vehicle center
    ray_depths = np.linalg.norm(pc.points[:3, :], axis=0)
    
    cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pc.translate(-np.array(cam_cs['translation']))
    pc.rotate(Quaternion(cam_cs['rotation']).rotation_matrix.T)
    
    depths_z = pc.points[2, :]
    intrinsic = np.array(cam_cs['camera_intrinsic'])
    pts_img = view_points(pc.points[:3, :], intrinsic, normalize=True)
    
    mask = (depths_z > 0.1) & (ray_depths > 1.0) & (ray_depths < 80.0)
    pts_img = pts_img[:, mask]
    ray_depths = ray_depths[mask]
    depths_z = depths_z[mask]
    
    u = np.round(pts_img[0, :]).astype(int)
    v = np.round(pts_img[1, :]).astype(int)
    
    width, height = cam_data['width'], cam_data['height']
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    
    u, v = u[valid], v[valid]
    ray_depths = ray_depths[valid]
    depths_z = depths_z[valid]
    
    order = np.argsort(-depths_z)
    u, v, ray_depths = u[order], v[order], ray_depths[order]
    
    gt_map = np.zeros((height, width), dtype=np.float32)
    gt_map[v, u] = ray_depths
    
    return gt_map

def compute_ray_depth_metrics(pred, gt):
    """Calculate Ray Depth metrics as defined in Appendix B."""
    mask = (gt > 1.0) & (gt < 80.0) & (pred > 0)
    if not mask.any():
        return None

    pred = pred[mask]
    gt = gt[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    delta_1 = (thresh < 1.25).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    return {"abs_rel": abs_rel, "delta_1": delta_1}

def main(args):
    if args.seq_len > 24:
        raise ValueError(f"Sequence length cannot exceed 24 frames as per the DVGT capacity. Received: {args.seq_len}")

    print(f"Loading official nuScenes devkit from {args.dataroot}...")
    nusc = NuScenes(version='v1.0-mini', dataroot=args.dataroot, verbose=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print("Initializing DVGT...")
    model = DVGT(frames_chunk_size=args.frames_chunk_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device).eval()

    all_pose_auc, all_depth_metrics = [], []
    total_inference_time = 0.0
    num_sequences = 0

    for scene in tqdm(nusc.scene, desc="Evaluating Scenes"):
        # Traverse scene keyframes
        sample_token = scene['first_sample_token']
        sample_tokens = []
        while sample_token != '':
            sample_tokens.append(sample_token)
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']

        # Evaluate on seq_len chunks to avoid GPU OOM
        sample_tokens = sample_tokens[:args.seq_len]
        T = len(sample_tokens)
        if T < 2: continue

        images, cam_img_sizes = [], []
        for token in sample_tokens:
            sample = nusc.get('sample', token)
            frame_images, sizes = [], []
            for cam in CAMERAS:
                cam_data = nusc.get('sample_data', sample['data'][cam])
                img_path = os.path.join(nusc.dataroot, cam_data['filename'])
                frame_images.append(preprocess_image(img_path))
                with Image.open(img_path) as img:
                    sizes.append(img.size)
            images.append(torch.stack(frame_images))
            cam_img_sizes.append(sizes)

        images_tensor = torch.stack(images).unsqueeze(0).to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad(), torch.amp.autocast(device, dtype=dtype):
            preds = model(images_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_inference_time += time.time() - start_time
        num_sequences += 1

        ego_n_to_ego_0 = pose_encoding_to_ego_pose(preds['ego_pose_enc'])
        ray_depth_in_ego_n = convert_point_in_ego_0_to_ray_depth_in_ego_n(preds['world_points'], ego_n_to_ego_0)

        pred_ego_poses = ego_n_to_ego_0[0].cpu().numpy()  # (T, 3, 4)
        pred_ray_depths = ray_depth_in_ego_n[0].cpu().numpy() # (T, V, H, W)

        pred_poses_44 = np.zeros((T, 4, 4))
        pred_poses_44[:, :3, :] = pred_ego_poses
        pred_poses_44[:, 3, 3] = 1.0

        # Calculate GT Poses (ego_n_to_ego_0)
        gt_poses_44 = []
        sample_0 = nusc.get('sample', sample_tokens[0])
        ego_pose_0 = nusc.get('ego_pose', nusc.get('sample_data', sample_0['data']['CAM_FRONT'])['ego_pose_token'])
        world_to_ego_0 = np.linalg.inv(get_transform_matrix(ego_pose_0['translation'], ego_pose_0['rotation']))

        for token in sample_tokens:
            sample = nusc.get('sample', token)
            ego_pose_n = nusc.get('ego_pose', nusc.get('sample_data', sample['data']['CAM_FRONT'])['ego_pose_token'])
            ego_n_to_world = get_transform_matrix(ego_pose_n['translation'], ego_pose_n['rotation'])
            gt_poses_44.append(world_to_ego_0 @ ego_n_to_world)
            
        gt_poses_44 = np.stack(gt_poses_44)

        # AUC@30 Pose Metric
        scene_auc = compute_pose_auc_30(pred_poses_44, gt_poses_44)
        all_pose_auc.append(scene_auc)

        # Depth Mapping
        for t, token in enumerate(sample_tokens):
            sample = nusc.get('sample', token)

            for v, cam in enumerate(CAMERAS):
                lidar_token = sample['data']['LIDAR_TOP']
                cam_token = sample['data'][cam]
                
                try:
                    gt_depth = project_ray_depth_to_image(nusc, lidar_token, cam_token)
                except Exception:
                    continue

                # Scale back to GT image dimensions and compute
                orig_w, orig_h = cam_img_sizes[t][v]
                pred_depth_crop = pred_ray_depths[t, v]
                pred_depth_resized = np.array(Image.fromarray(pred_depth_crop).resize((orig_w, orig_h), Image.Resampling.NEAREST))

                metrics = compute_ray_depth_metrics(pred_depth_resized, gt_depth)
                if metrics:
                    all_depth_metrics.append(metrics)

    print("\n" + "=" * 50)
    print("DVGT EVALUATION RESULTS (nuScenes Mini)")
    print("=" * 50)
    print(f"Pose AUC@30: {np.mean(all_pose_auc):.2f}")
    print("-" * 50)
    print("Ray Depth Metrics:")
    for k in ['abs_rel', 'delta_1']:
        print(f"  {k:8s}: {np.mean([m[k] for m in all_depth_metrics]):.4f}")

    print("-" * 50)
    print("Profiling Metrics:")
    if num_sequences > 0:
        avg_time = total_inference_time / num_sequences
        fps = (args.seq_len * num_sequences) / total_inference_time
        print(f"  Avg Inference Time (per {args.seq_len}-frame seq): {avg_time:.4f} s")
        print(f"  Effective Inference FPS: {fps:.2f}")
    if torch.cuda.is_available():
        peak_vram_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_vram_reserv = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        print(f"  Peak VRAM Allocated: {peak_vram_alloc:.2f} MB")
        print(f"  Peak VRAM Reserved:  {peak_vram_reserv:.2f} MB")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default=os.path.expanduser('~/datasets/nuscenes-mini'))
    parser.add_argument('--checkpoint', type=str, default='ckpt/open_ckpt.pt')
    parser.add_argument('--seq_len', type=int, default=16, help="Frames to evaluate per scene (helps prevent OOM).")
    parser.add_argument('--frames_chunk_size', type=int, default=8)
    args = parser.parse_args()
    main(args)