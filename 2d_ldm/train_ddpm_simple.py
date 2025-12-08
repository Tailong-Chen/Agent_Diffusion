"""
DDPM训练脚本（纯FP32稳定版，无AutoEncoder）
修改内容：
1. 移除了混合精度(autocast)，解决训练后期生成全黑/NaN的问题。
2. 每10个epoch自动保存模型。
3. 采样时增加数值范围打印，便于监控。

使用方法:
    python train_ddpm_simple.py --tiff_path ./data/mt.tif --output_dir ./output_ddpm_256 --image_size 256 --batch_size 4
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
# 移除 AMP 相关的引用，确保数值稳定
# from torch.cuda.amp import autocast, GradScaler 
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile

from monai import transforms
from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


class TiffDataset:
    """TIFF数据集"""
    
    def __init__(self, images_array, transform=None):
        self.data = [{"image": img} for img in images_array]
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]["image"].astype(np.float32)
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        data_dict = {"image": image}
        if self.transform:
            data_dict = self.transform(data_dict)
        return data_dict


def train_ddpm(
    tiff_path,
    output_dir,
    image_size=256,
    n_epochs=150,
    batch_size=4,
    val_interval=200,
    sample_interval=200, # 建议设置小一点，比如 20或50，方便观察
    num_inference_steps=1000,
    seed=42,
    resume_from=None,
    load_optimizer=False,
):
    """
    DDPM训练主函数 (FP32稳定版)
    """
    
    # ============================================
    # 1. 设置
    # ============================================
    set_determinism(seed)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)
    
    print("="*70)
    print("🚀 DDPM训练（FP32稳定版 - 解决纯黑Bug）")
    print("="*70)
    print(f"图像尺寸: {image_size}×{image_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {n_epochs}")
    print(f"每10个Epoch保存一次模型")
    
    # ============================================
    # 2. 加载数据
    # ============================================
    print("\n📂 加载数据...")
    all_images = tifffile.imread(tiff_path)
    if all_images.ndim == 2:
        all_images = all_images[np.newaxis, ...]
    
    print(f"总图像数: {len(all_images)}")
    
    # 确定归一化范围
    if all_images.dtype == np.uint8:
        scale_max = 255.0
    elif all_images.dtype == np.uint16:
        scale_max = 65535.0
    else:
        scale_max = float(all_images.max())
    
    print(f"数据范围: [0, {scale_max}]")
    
    # 划分数据集
    train_size = int(0.9 * len(all_images))
    indices = np.random.permutation(len(all_images))
    train_images = all_images[indices[:train_size]]
    val_images = all_images[indices[train_size:]]
    
    # ============================================
    # 3. 数据变换
    # ============================================
    train_transforms = transforms.Compose([
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=scale_max,
            b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.RandAffined(
            keys=["image"],
            rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
            translate_range=[(-1, 1), (-1, 1)],
            scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
            spatial_size=[image_size, image_size],
            padding_mode="zeros",
            prob=0.5,
        ),
    ])
    
    val_transforms = transforms.Compose([
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=scale_max,
            b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.Resized(
            keys=["image"],
            spatial_size=[image_size, image_size],
        ),
    ])
    
    # 创建数据集
    train_ds = CacheDataset(
        data=TiffDataset(train_images, transform=None).data,
        transform=train_transforms
    )
    val_ds = CacheDataset(
        data=TiffDataset(val_images, transform=None).data,
        transform=val_transforms
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, persistent_workers=True
    )
    
    # ============================================
    # 4. 初始化模型
    # ============================================
    print("\n🔧 初始化UNet...")
    
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
    
    model_params = sum(p.numel() for p in unet.parameters()) / 1e6
    print(f"✅ UNet参数量: {model_params:.2f}M")
    
    # Scheduler和Inferer
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    inferer = DiffusionInferer(scheduler)
    
    # 优化器
    optimizer = torch.optim.Adam(params=unet.parameters(), lr=1e-4)
    
    # ⚠️ 已移除 GradScaler，因为我们现在使用纯 FP32 训练
    
    start_epoch = 0

    if resume_from is not None:
        ckpt_path = Path(resume_from)
        if ckpt_path.exists():
            print(f"🔁 加载checkpoint: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=device)
            
            if 'model_state_dict' in ckpt:
                unet.load_state_dict(ckpt['model_state_dict'])
                print("   ✅ 模型权重已加载")
            
            if load_optimizer and 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    print("   ✅ 优化器状态已加载")
                except Exception as e:
                    print(f"   ❌ 无法加载优化器状态: {e}")

            start_epoch = int(ckpt.get('epoch', 0))
            print(f"   ▶️ 从 epoch {start_epoch+1} 开始继续训练")
        else:
            print(f"⚠️ 指定的 checkpoint 不存在: {ckpt_path}，将从头开始训练")
    
    # ============================================
    # 5. 训练循环
    # ============================================
    print("\n🚀 开始训练 (FP32模式)...\n")
    
    best_val_loss = float('inf')
    epoch_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, n_epochs):
        # ========== 训练 ==========
        unet.train()
        epoch_loss = 0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{n_epochs}",
            ncols=110
        )
        
        for batch in progress_bar:
            images = batch["image"].to(device)
            
            # 确保是 float32
            images = images.float()
            
            optimizer.zero_grad(set_to_none=True)
            
            # ❌ 移除 autocast
            # with autocast(enabled=True):
            
            # 生成噪声
            noise = torch.randn_like(images).to(device)
            
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps,
                (images.shape[0],),
                device=device
            ).long()
            
            # 前向传播
            noise_pred = inferer(
                inputs=images,
                diffusion_model=unet,
                noise=noise,
                timesteps=timesteps
            )
            
            loss = F.mse_loss(noise_pred, noise)
            
            # ❌ 移除 scaler，使用标准反向传播
            loss.backward()
            optimizer.step()
            # scaler.update()
            
            epoch_loss += loss.item()
            
            n_batches = progress_bar.n if progress_bar.n > 0 else 1
            progress_bar.set_postfix({"loss": f"{epoch_loss/n_batches:.5f}"})
        
        avg_train_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_train_loss)
        
        # ========== 验证 ==========
        if (epoch + 1) % val_interval == 0:
            unet.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device).float() # 确保 float32
                    
                    noise = torch.randn_like(images).to(device)
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps,
                        (images.shape[0],),
                        device=device
                    ).long()
                    
                    noise_pred = inferer(
                        inputs=images,
                        diffusion_model=unet,
                        noise=noise,
                        timesteps=timesteps
                    )
                    loss = F.mse_loss(noise_pred, noise)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            print(f"\n📊 Epoch {epoch+1} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                }, output_dir / "checkpoints" / "best_model.pth")
                print("💾 Best model saved!")

        # ========== 每10个Epoch保存一次模型 (新增) ==========
        if (epoch + 1) % 10 == 0:
            save_path = output_dir / "checkpoints" / f"model_epoch_{epoch+1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": unet.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
            }, save_path)
            print(f"💾 Checkpoint saved: {save_path.name}")

        # ========== 生成样本（带进度条 + 数值监控） ==========
        if (epoch + 1) % sample_interval == 0:
            print(f"   🎨 生成样本中（{num_inference_steps}步）...")
            unet.eval()
            scheduler.set_timesteps(num_inference_steps=num_inference_steps)
            
            with torch.no_grad():
                # 1. 初始化噪声 (确保是float32)
                noise = torch.randn((1, 1, image_size, image_size)).to(device).float()
                image = noise
                
                progress_sampling = tqdm(
                    scheduler.timesteps,
                    desc="   Sampling",
                    ncols=110,
                    leave=False
                )
                
                # ❌ 必须移除 autocast，否则采样容易变成全黑
                for t in progress_sampling:
                    model_output = unet(
                        x=image,
                        timesteps=torch.Tensor((t,)).to(device).long()
                    )
                    step_result = scheduler.step(model_output, t, image)
                    
                    if isinstance(step_result, tuple):
                        image = step_result[0]
                    else:
                        image = step_result.prev_sample
            
            # 2. 数值监控 (打印出来看看是不是全是0或者Nan)
            d_min, d_max, d_mean = image.min().item(), image.max().item(), image.mean().item()
            print(f"   🔍 样本统计 - Min: {d_min:.4f} | Max: {d_max:.4f} | Mean: {d_mean:.4f}")
            
            if d_max == 0 and d_min == 0:
                print("   ❌ 警告：生成的图像是纯黑（全0）！")
            if np.isnan(d_mean):
                print("   ❌ 警告：生成的图像包含 NaN！")

            # 保存样本
            plt.figure(figsize=(6, 6))
            plt.imshow(image[0, 0].cpu().numpy(), vmin=0, vmax=1, cmap="gray")
            plt.title(f"Epoch {epoch+1} (Max:{d_max:.2f})")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(
                output_dir / "samples" / f"sample_epoch_{epoch+1}.png",
                dpi=150, bbox_inches='tight'
            )
            plt.close()
            print(f"   ✅ 样本已保存\n")
    
    # ============================================
    # 6. 训练完成后的处理
    # ============================================
    print("\n" + "="*70)
    print("✅ 训练完成！")
    print("="*70)
    
    # 绘制学习曲线
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label="Train Loss", linewidth=2)
    if val_losses:
        val_epochs = np.arange(val_interval - 1, n_epochs, val_interval)
        plt.plot(val_epochs, val_losses, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=150)
    plt.close()
    
    # 生成最终样本网格
    print("\n🎨 生成最终样本网格（8张）...")
    unet.eval()
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i in range(8):
            noise = torch.randn((1, 1, image_size, image_size)).to(device).float()
            image = noise
            
            progress_sampling = tqdm(
                scheduler.timesteps,
                desc=f"   Sample {i+1}",
                ncols=110,
                leave=False
            )
            
            for t in progress_sampling:
                model_output = unet(
                    x=image,
                    timesteps=torch.Tensor((t,)).to(device).long()
                )
                step_result = scheduler.step(model_output, t, image)
                if isinstance(step_result, tuple):
                    image = step_result[0]
                else:
                    image = step_result.prev_sample
            
            axes[i].imshow(image[0, 0].cpu().numpy(), vmin=0, vmax=1, cmap="gray")
            axes[i].set_title(f"Sample {i+1}")
            axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig(output_dir / "final_samples.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 全部完成: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="简单DDPM训练（稳定版，无AMP）")
    parser.add_argument("--tiff_path", type=str, required=True, help="TIFF文件路径")
    parser.add_argument("--output_dir", type=str, default="./output_ddpm", help="输出目录")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--n_epochs", type=int, default=150, help="训练轮数")
    parser.add_argument("--val_interval", type=int, default=200,help="验证间隔")
    parser.add_argument("--sample_interval", type=int, default=10, help="采样间隔（推荐设置小一点以监控）")
    parser.add_argument("--num_inference_steps", type=int, default=1000, help="采样步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume_from", type=str, default=None, help="checkpoint路径")
    parser.add_argument("--load_optimizer", action="store_true", help="是否加载优化器")
    
    args = parser.parse_args()
    
    train_ddpm(
        tiff_path=args.tiff_path,
        output_dir=args.output_dir,
        image_size=args.image_size,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        val_interval=args.val_interval,
        sample_interval=args.sample_interval,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        resume_from=args.resume_from,
        load_optimizer=args.load_optimizer,
    )