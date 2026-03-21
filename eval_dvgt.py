import argparse
import os
import torch
from dvgt.models.dvgt import DVGT
from dvgt.utils.load_fn import load_and_preprocess_images
from iopath.common.file_io import g_pathmgr
from nuscenes.nuscenes import NuScenes
from PIL import Image
from torchvision import transforms as TF
from demo_viser import visualize_pred, launch_viser_server

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
            predictions = model(images_tensor)

    vis_args = argparse.Namespace(
        conf_threshold=25.0,
        mask_sky=False,
        use_edge_masks=False,
        edge_depth_rtol=0.1,
        edge_normal_tol=50,
        max_depth=-1,
        downsample_ratio=-1,
    )
    point_clouds, poses = visualize_pred(predictions, vis_args)
    launch_viser_server(point_clouds, poses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default=os.path.expanduser('~/datasets/nuscenes-mini'))
    # parser.add_argument('--checkpoint', type=str, default='ckpt/open_ckpt.pt')
    parser.add_argument('--frames', type=int, default=16, choices=range(1, 25), help="Frames to evaluate per scene (Max 24).")
    # parser.add_argument('--frames_chunk_size', type=int, default=8)
    args = parser.parse_args()
    main(args)