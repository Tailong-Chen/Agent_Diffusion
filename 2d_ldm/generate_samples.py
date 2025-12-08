"""
使用训练好的LDM生成新样本
支持批量生成和可视化

使用方法:
    python generate_samples.py --checkpoint ./output_ldm/checkpoints/diffusion_epoch_250.pth --num_samples 10
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from torch.cuda.amp import autocast

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import LatentDiffusionInferer

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    print("⚠️  未安装tifffile，无法保存TIFF格式")


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    print(f"📦 加载checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取配置
    config = checkpoint.get("config", {})
    
    # 重建AutoencoderKL配置
    if "autoencoder_config" in config:
        ae_config = config["autoencoder_config"]
    else:
        # 默认配置
        ae_config = {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": (128, 256, 512),
            "latent_channels": 4,
            "num_res_blocks": 2,
            "attention_levels": (False, False, True),
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
        }
    
    # 重建UNet配置
    if "unet_config" in config:
        unet_config = config["unet_config"]
    else:
        unet_config = {
            "spatial_dims": 2,
            "in_channels": 4,
            "out_channels": 4,
            "num_res_blocks": 2,
            "num_channels": (128, 256, 512, 768),
            "attention_levels": (False, False, True, True),
            "num_head_channels": (0, 0, 512, 768),
        }
    
    # 创建模型
    print("🔧 创建模型...")
    autoencoder = AutoencoderKL(**ae_config).to(device)
    unet = DiffusionModelUNet(**unet_config).to(device)
    
    # 加载权重
    autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
    unet.load_state_dict(checkpoint["unet_state_dict"])
    
    # 设置为评估模式
    autoencoder.eval()
    unet.eval()
    
    # 获取scale_factor
    scale_factor = checkpoint.get("scale_factor", 1.0)
    
    # 获取scheduler配置
    if "scheduler_config" in config:
        scheduler_config = config["scheduler_config"]
    else:
        scheduler_config = {
            "num_train_timesteps": 1000,
            "schedule": "scaled_linear_beta",
            "beta_start": 0.00085,
            "beta_end": 0.012,
        }
    
    scheduler = DDPMScheduler(**scheduler_config)
    
    print(f"✅ 模型加载完成")
    print(f"   - Scaling factor: {scale_factor:.4f}")
    print(f"   - Latent channels: {ae_config['latent_channels']}")
    
    # 计算潜在空间尺寸
    image_size = config.get("image_size", 1024)
    num_layers = len(ae_config["num_channels"])
    latent_size = image_size // (2 ** num_layers)
    
    return autoencoder, unet, scheduler, scale_factor, ae_config["latent_channels"], latent_size


def generate_samples(
    autoencoder,
    unet,
    scheduler,
    scale_factor,
    latent_channels,
    latent_size,
    device,
    num_samples=10,
    num_inference_steps=1000,
    save_intermediates=False,
    batch_size=1,
):
    """生成样本"""
    
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    print(f"\n🎨 生成 {num_samples} 个样本...")
    print(f"   - 推理步数: {num_inference_steps}")
    print(f"   - 潜在空间: {latent_size}×{latent_size}×{latent_channels}")
    
    all_samples = []
    all_intermediates = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="生成中"):
            current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
            
            # 从噪声开始
            noise = torch.randn(
                (current_batch_size, latent_channels, latent_size, latent_size)
            ).to(device)
            
            with autocast(enabled=True):
                if save_intermediates:
                    samples, intermediates = inferer.sample(
                        input_noise=noise,
                        diffusion_model=unet,
                        scheduler=scheduler,
                        autoencoder_model=autoencoder,
                        save_intermediates=True,
                        intermediate_steps=num_inference_steps // 10,
                    )
                    all_intermediates.append(intermediates)
                else:
                    samples = inferer.sample(
                        input_noise=noise,
                        diffusion_model=unet,
                        scheduler=scheduler,
                        autoencoder_model=autoencoder,
                    )
            
            all_samples.append(samples.cpu())
    
    all_samples = torch.cat(all_samples, dim=0)
    
    print(f"✅ 生成完成！形状: {all_samples.shape}")
    
    if save_intermediates:
        return all_samples, all_intermediates
    return all_samples, None


def save_samples(samples, output_dir, prefix="sample"):
    """保存生成的样本"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存样本到: {output_dir}")
    
    # 保存为单独的图像
    for i, sample in enumerate(samples):
        img = sample[0].numpy()  # (H, W)
        
        # 保存为PNG
        plt.figure(figsize=(10, 10))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.tight_layout(pad=0)
        png_path = output_dir / f"{prefix}_{i+1:04d}.png"
        plt.savefig(png_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
    
    print(f"✅ 保存了 {len(samples)} 个PNG文件")
    
    # 保存为TIFF堆栈
    if HAS_TIFFFILE and len(samples) > 1:
        tiff_stack = np.stack([s[0].numpy() for s in samples])
        tiff_path = output_dir / f"{prefix}_stack.tif"
        tifffile.imwrite(tiff_path, tiff_stack)
        print(f"✅ 保存TIFF堆栈: {tiff_path}")
    
    # 创建网格可视化
    num_samples = len(samples)
    ncols = min(5, num_samples)
    nrows = (num_samples + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]
    
    for idx in range(num_samples):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].imshow(samples[idx, 0].numpy(), cmap='gray')
        axes[row][col].set_title(f"Sample {idx+1}")
        axes[row][col].axis('off')
    
    # 隐藏多余的子图
    for idx in range(num_samples, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis('off')
    
    plt.tight_layout()
    grid_path = output_dir / f"{prefix}_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 保存网格图: {grid_path}")


def visualize_denoising_process(intermediates, output_path):
    """可视化去噪过程"""
    print(f"\n🎬 可视化去噪过程...")
    
    # 取第一个样本
    if isinstance(intermediates, list):
        intermediates = intermediates[0]
    
    num_steps = len(intermediates)
    
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps*3, 3))
    if num_steps == 1:
        axes = [axes]
    
    for i, intermediate in enumerate(intermediates):
        img = intermediate[0, 0].cpu().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Step {i*100}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 去噪过程已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="使用LDM生成新样本")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型checkpoint路径")
    parser.add_argument("--output_dir", type=str, default="./generated_samples", help="输出目录")
    parser.add_argument("--num_samples", type=int, default=10, help="生成样本数量")
    parser.add_argument("--num_inference_steps", type=int, default=1000, help="推理步数")
    parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
    parser.add_argument("--save_intermediates", action="store_true", help="保存中间去噪步骤")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 检查CUDA
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  设备: {device}")
    
    # 加载模型
    autoencoder, unet, scheduler, scale_factor, latent_channels, latent_size = load_model(
        args.checkpoint, device
    )
    
    # 生成样本
    samples, intermediates = generate_samples(
        autoencoder,
        unet,
        scheduler,
        scale_factor,
        latent_channels,
        latent_size,
        device,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        save_intermediates=args.save_intermediates,
        batch_size=args.batch_size,
    )
    
    # 保存样本
    save_samples(samples, args.output_dir)
    
    # 可视化去噪过程
    if args.save_intermediates and intermediates:
        denoising_path = Path(args.output_dir) / "denoising_process.png"
        visualize_denoising_process(intermediates, denoising_path)
    
    print("\n" + "="*60)
    print("🎉 生成完成！")
    print("="*60)
    print(f"📁 输出目录: {args.output_dir}")
    print(f"   - sample_XXXX.png: 单独的样本")
    print(f"   - sample_grid.png: 样本网格")
    if HAS_TIFFFILE:
        print(f"   - sample_stack.tif: TIFF堆栈")
    if args.save_intermediates:
        print(f"   - denoising_process.png: 去噪过程")
    print("="*60)


if __name__ == "__main__":
    main()

