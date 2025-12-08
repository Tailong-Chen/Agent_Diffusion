"""
DDPM采样脚本
用于加载训练好的DDPM模型权重并生成新样本。

支持自动查找最佳权重或最新权重。
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from pathlib import Path
from tqdm import tqdm
from monai.utils import set_determinism
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

def find_best_checkpoint(model_dir):
    """查找最佳或最新的checkpoint"""
    model_dir = Path(model_dir)
    ckpt_dir = model_dir / "checkpoints"
    
    if not ckpt_dir.exists():
        # 尝试直接在model_dir下查找
        ckpt_dir = model_dir
    
    # 1. 优先查找 best_model.pth
    best_model = ckpt_dir / "best_model.pth"
    if best_model.exists():
        print(f"Found best model: {best_model}")
        return best_model
    
    # 2. 查找最新的 model_epoch_*.pth
    ckpts = list(ckpt_dir.glob("model_epoch_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    
    # 按epoch排序
    def get_epoch(p):
        try:
            return int(p.stem.split("_")[-1])
        except:
            return 0
            
    latest_ckpt = sorted(ckpts, key=get_epoch)[-1]
    print(f"Found latest model: {latest_ckpt}")
    return latest_ckpt

def sample_ddpm(
    model_dir,
    output_dir,
    image_size=512,
    num_samples=10,
    batch_size=1,
    num_inference_steps=1000,
    seed=42,
    device_name="cuda:0"
):
    set_determinism(seed)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    
    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir / "generated_samples"
    else:
        output_dir = Path(output_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Model Dir: {model_dir}")
    print(f"Output Dir: {output_dir}")
    print(f"Image Size: {image_size}")
    print(f"Device: {device}")
    
    # ============================================
    # 1. 初始化模型
    # ============================================
    if image_size >= 512:
        num_channels = (128, 256, 512)
        attention_levels = (False, False, False)
        num_head_channels = 512
    else:
        num_channels = (128, 256, 256)
        attention_levels = (False, True, True)
        num_head_channels = 256
    
    unet = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_res_blocks=2,
        num_channels=num_channels,
        attention_levels=attention_levels,
        num_head_channels=num_head_channels,
    ).to(device)
    
    # ============================================
    # 2. 加载权重
    # ============================================
    ckpt_path = find_best_checkpoint(model_dir)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    
    if 'model_state_dict' in ckpt:
        unet.load_state_dict(ckpt['model_state_dict'])
    else:
        unet.load_state_dict(ckpt)
    
    print("✅ Model loaded successfully")
    unet.eval()
    
    # ============================================
    # 3. 采样
    # ============================================
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    inferer = DiffusionInferer(scheduler)
    
    generated_images = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Starting sampling: {num_samples} images in {num_batches} batches")
    
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            noise = torch.randn((current_batch_size, 1, image_size, image_size)).to(device)
            scheduler.set_timesteps(num_inference_steps=num_inference_steps)
            
            image = noise
            for t in tqdm(scheduler.timesteps, desc=f"Batch {i+1}/{num_batches}"):
                model_output = unet(
                    x=image,
                    timesteps=torch.Tensor((t,)).to(device).long()
                )
                step_result = scheduler.step(model_output, t, image)
                
                if isinstance(step_result, tuple):
                    image = step_result[0]
                else:
                    image = step_result.prev_sample
            
            # 收集结果
            generated_images.append(image.cpu().numpy())
            
            # 保存单个批次的图像 (纯净TIFF)
            for j in range(current_batch_size):
                img_idx = i * batch_size + j
                img_data = image[j, 0].cpu().numpy().astype(np.float32)
                
                save_path = output_dir / f"sample_{img_idx:04d}.tif"
                tifffile.imwrite(save_path, img_data)

    # ============================================
    # 4. 保存结果
    # ============================================
    all_images = np.concatenate(generated_images, axis=0) # (N, 1, H, W)
    all_images = all_images[:, 0, :, :] # (N, H, W)
    
    # 保存为TIFF堆栈
    tiff_path = output_dir / "generated_stack.tif"
    tifffile.imwrite(tiff_path, all_images.astype(np.float32))
    print(f"✅ Saved TIFF stack to {tiff_path}")
    
    # 打印统计信息
    print(f"Stats - Min: {all_images.min():.4f}, Max: {all_images.max():.4f}, Mean: {all_images.mean():.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="包含checkpoints的目录")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--image_size", type=int, default=512, help="图像尺寸")
    parser.add_argument("--num_samples", type=int, default=10, help="生成样本数量")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--steps", type=int, default=1000, help="采样步数")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    
    args = parser.parse_args()
    
    sample_ddpm(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_inference_steps=args.steps,
        device_name=args.device
    )
