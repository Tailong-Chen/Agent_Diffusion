import argparse
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
from pathlib import Path
import copy

import util.misc as misc
from denoiser_unet import DenoiserUNet as Denoiser

def get_args_parser():
    parser = argparse.ArgumentParser('JiT-UNet Sampling', add_help=False)

    # Architecture args (must match training)
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--in_channels', default=1, type=int, help='Input channels')
    parser.add_argument('--out_channels', default=1, type=int, help='Output channels')
    parser.add_argument('--class_num', default=1, type=int)
    
    # Model specific args
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--ema_decay1', type=float, default=0.9999)
    parser.add_argument('--ema_decay2', type=float, default=0.9996)

    # Sampling args
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output_dir', default='./generated_samples', help='Directory to save outputs')
    parser.add_argument('--sampling_method', default='heun', type=str, help='ODE sampling method')
    parser.add_argument('--num_sampling_steps', default=50, type=int, help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float, help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float, help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float, help='CFG interval max')
    parser.add_argument('--num_images', default=16, type=int, help='Number of images to generate')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for generation')
    parser.add_argument('--target_img_size', default=None, type=int, help='Target image size for generation (e.g. 1024). If None, uses model training size.')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda', help='Device to use')

    # Distributed args
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='URL used to set up distributed training')

    return parser

def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Create model
    model = Denoiser(args)
    model.to(device)
    
    # Load checkpoint
    if os.path.isfile(args.ckpt):
        print(f"Loading checkpoint from {args.ckpt}")
        checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            
            # Try to load EMA weights if available and desired
            # Usually for sampling we want EMA weights. 
            # The training script stores them in 'model_ema1' and 'model_ema2'
            # But here we just want to load one set of weights into the model for sampling.
            
            # Let's check if we should use EMA. 
            # Typically EMA weights are better.
            if 'model_ema1' in checkpoint:
                print("Loading EMA1 weights for sampling...")
                ema_state_dict = checkpoint['model_ema1']
                # The EMA dict keys might match the model keys directly
                # In training: model_without_ddp.ema_params1 is a list of params.
                # Wait, let's check how checkpoint is saved in engine_jit.py or misc.save_model
                
                # In main_jit_unet.py:
                # model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
                
                # It seems 'model_ema1' in checkpoint is a state_dict (name -> tensor).
                # So we can just load it.
                msg = model.load_state_dict(ema_state_dict, strict=False)
                print(f"EMA weights loaded with msg: {msg}")
            else:
                print("No EMA weights found, using standard weights.")
        else:
            # Maybe the checkpoint is just the state dict
            model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Checkpoint not found: {args.ckpt}")

    model.eval()

    # Setup output directory
    if misc.get_rank() == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving images to {args.output_dir}")

    # Generation loop
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    
    # Ensure num_images is divisible by world_size for simplicity
    if args.num_images % world_size != 0:
        args.num_images = (args.num_images // world_size + 1) * world_size
    
    num_images_per_rank = args.num_images // world_size
    num_batches = (num_images_per_rank + args.batch_size - 1) // args.batch_size
    
    target_size = args.target_img_size if args.target_img_size is not None else args.img_size
    if local_rank == 0:
        print(f"Target generation size: {target_size}x{target_size}")
        if target_size > args.img_size:
            print(f"Using Overlap-Tile Sampling (Model: {args.img_size} -> Target: {target_size})")
    
    print(f"Rank {local_rank}: Generating {num_images_per_rank} images in {num_batches} batches")

    # Class labels
    # Assuming class 0 for now if class_num=1
    # If class_num > 1, we might want to sample specific classes or random ones.
    # For now, let's generate class 0.
    
    cnt = 0
    for i in range(num_batches):
        current_batch_size = min(args.batch_size, num_images_per_rank - cnt)
        if current_batch_size <= 0:
            break
            
        labels = torch.zeros(current_batch_size, device=device).long()
        # If we want random classes:
        if args.class_num > 1:
            labels = torch.randint(0, args.class_num, (current_batch_size,), device=device).long()

        print(f"Rank {local_rank}: Batch {i+1}/{num_batches}, size {current_batch_size}")
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                # Pass target_img_size to generate method
                sampled_images = model.generate(labels, img_size=target_size)

        # Denormalize
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.clamp(0, 1).cpu().numpy()

        # Save images
        for b_id in range(current_batch_size):
            img_idx = local_rank * num_images_per_rank + cnt + b_id
            
            # Handle channels
            img = sampled_images[b_id] # C, H, W
            
            if img.shape[0] == 1:
                # Grayscale
                img = (img[0] * 255).astype(np.uint8)
                # cv2 expects H, W for grayscale or H, W, 3 for color
            else:
                # RGB or other
                img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            save_path = os.path.join(args.output_dir, f"sample_{img_idx:05d}.png")
            cv2.imwrite(save_path, img)
            
        cnt += current_batch_size

    if args.distributed:
        torch.distributed.barrier()
    print("Done!")

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
