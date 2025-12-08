"""
多卡DDPM训练脚本（支持DDP）
基于官方教程，支持多GPU并行训练

使用方法:
    # 单卡
    python train_ddpm_tiff_ddp.py --tiff_path ./data/mt.tif --image_size 512
    
    # 多卡（例如4卡）
    torchrun --nproc_per_node=4 train_ddpm_tiff_ddp.py \
        --tiff_path ./data/mt.tif \
        --image_size 512 \
        --batch_size 2  # 每卡batch=2，总batch=2*4=8
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import tifffile
import time
import os

from monai import transforms
from monai.utils import set_determinism
from monai.data import CacheDataset, DataLoader

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


def setup_ddp():
    """初始化DDP"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 单卡模式
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl')
    
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_ddp():
    """清理DDP"""
    if dist.is_initialized():
        dist.destroy_process_group()


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


def main():
    parser = argparse.ArgumentParser(description="多卡DDPM训练")
    parser.add_argument("--tiff_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output_ddpm_ddp")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2, 
                       help="每卡的batch size（总batch = batch_size × GPU数）")
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # ============================================
    # 初始化DDP
    # ============================================
    rank, world_size, local_rank = setup_ddp()
    is_main_process = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process:
        print("="*70)
        print(f"🚀 多卡DDPM训练")
        print("="*70)
        print(f"GPU数量: {world_size}")
        print(f"图像尺寸: {args.image_size}×{args.image_size}")
        print(f"每卡Batch: {args.batch_size}")
        print(f"总Batch: {args.batch_size * world_size}")
        print(f"训练轮数: {args.n_epochs}")
    
    set_determinism(args.seed)
    
    # 创建输出目录（只在主进程）
    if is_main_process:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        (output_dir / "samples").mkdir(exist_ok=True)
    
    # ============================================
    # 加载数据
    # ============================================
    if is_main_process:
        print("\n📂 加载数据...")
    
    all_images = tifffile.imread(args.tiff_path)
    if all_images.ndim == 2:
        all_images = all_images[np.newaxis, ...]
    
    # 确定归一化范围
    if all_images.dtype == np.uint8:
        scale_max = 255.0
    elif all_images.dtype == np.uint16:
        scale_max = 65535.0
    else:
        scale_max = float(all_images.max())
    
    # 划分数据
    train_size = int(0.9 * len(all_images))
    indices = np.random.permutation(len(all_images))
    train_images = all_images[indices[:train_size]]
    val_images = all_images[indices[train_size:]]
    
    if is_main_process:
        print(f"   训练集: {len(train_images)} 张")
        print(f"   验证集: {len(val_images)} 张")
        print(f"   数据范围: [0, {scale_max}]")
    
    # ============================================
    # 数据变换
    # ============================================
    train_transforms = transforms.Compose([
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=scale_max,
            b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.Resized(
            keys=["image"], 
            spatial_size=[args.image_size, args.image_size]
        ),
        transforms.RandAffined(
            keys=["image"],
            rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
            translate_range=[(-1, 1), (-1, 1)],
            scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
            spatial_size=[args.image_size, args.image_size],
            padding_mode="zeros",
            prob=0.5,
        ),
    ])
    
    val_transforms = transforms.Compose([
        transforms.ScaleIntensityRanged(
            keys=["image"], a_min=0.0, a_max=scale_max,
            b_min=0.0, b_max=1.0, clip=True
        ),
        transforms.Resized(
            keys=["image"], 
            spatial_size=[args.image_size, args.image_size]
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
    
    # 创建DistributedSampler
    train_sampler = DistributedSampler(
        train_ds, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True,
        seed=args.seed
    ) if world_size > 1 else None
    
    val_sampler = DistributedSampler(
        val_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    ) if world_size > 1 else None
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # ============================================
    # 初始化模型
    # ============================================
    if is_main_process:
        print("\n🔧 初始化模型...")
    
    # 根据图像尺寸调整配置
    if args.image_size <= 128:
        num_channels = (128, 256, 256)
        attention_levels = (False, True, True)
        num_head_channels = 256
    elif args.image_size == 256:
        num_channels = (128, 256, 512)
        attention_levels = (False, True, True)
        num_head_channels = 512
    else:  # 512+
        num_channels = (128, 256, 512)
        attention_levels = (False, False, True)
        num_head_channels = 512
    
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=num_channels,
        attention_levels=attention_levels,
        num_res_blocks=1,
        num_head_channels=num_head_channels,
    ).to(device)
    
    # 包装为DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if is_main_process:
        model_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"✅ 模型参数量: {model_params:.2f}M")
    
    # Scheduler和Optimizer
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 学习率根据GPU数量缩放（可选）
    base_lr = 2.5e-5
    lr = base_lr * world_size  # 线性缩放
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    
    inferer = DiffusionInferer(scheduler)
    
    # ============================================
    # 训练
    # ============================================
    if is_main_process:
        print("\n🚀 开始训练...\n")
    
    epoch_loss_list = []
    val_epoch_loss_list = []
    scaler = GradScaler()
    total_start = time.time()
    
    for epoch in range(args.n_epochs):
        model.train()
        
        # 设置sampler的epoch（用于shuffle）
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        epoch_loss = 0
        
        # 只在主进程显示进度条
        if is_main_process:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
            progress_bar.set_description(f"Epoch {epoch}")
        else:
            progress_bar = enumerate(train_loader)
        
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=True):
                noise = torch.randn_like(images).to(device)
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps,
                    (images.shape[0],),
                    device=images.device
                ).long()
                
                noise_pred = inferer(
                    inputs=images,
                    diffusion_model=model,
                    noise=noise,
                    timesteps=timesteps
                )
                
                loss = F.mse_loss(noise_pred.float(), noise.float())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            if is_main_process:
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # 收集所有GPU的loss（平均）
        if world_size > 1:
            avg_train_loss_tensor = torch.tensor(avg_train_loss).to(device)
            dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.AVG)
            avg_train_loss = avg_train_loss_tensor.item()
        
        if is_main_process:
            epoch_loss_list.append(avg_train_loss)
        
        # ============================================
        # 验证
        # ============================================
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            
            for step, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                with torch.no_grad():
                    with autocast(enabled=True):
                        noise = torch.randn_like(images).to(device)
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps,
                            (images.shape[0],),
                            device=images.device
                        ).long()
                        noise_pred = inferer(
                            inputs=images,
                            diffusion_model=model,
                            noise=noise,
                            timesteps=timesteps
                        )
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())
                
                val_epoch_loss += val_loss.item()
            
            avg_val_loss = val_epoch_loss / len(val_loader)
            
            # 收集所有GPU的val loss
            if world_size > 1:
                avg_val_loss_tensor = torch.tensor(avg_val_loss).to(device)
                dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.AVG)
                avg_val_loss = avg_val_loss_tensor.item()
            
            if is_main_process:
                val_epoch_loss_list.append(avg_val_loss)
                print(f"\n📊 Epoch {epoch+1}/{args.n_epochs} | "
                      f"Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")
                
                # 保存checkpoint
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                }, Path(args.output_dir) / "checkpoints" / f"model_epoch_{epoch+1}.pth")
                
                # 生成样本
                noise = torch.randn((1, 1, args.image_size, args.image_size)).to(device)
                scheduler.set_timesteps(num_inference_steps=1000)
                
                with autocast(enabled=True):
                    # 使用model.module如果是DDP
                    sample_model = model.module if world_size > 1 else model
                    image = inferer.sample(
                        input_noise=noise,
                        diffusion_model=sample_model,
                        scheduler=scheduler
                    )
                
                plt.figure(figsize=(6, 6))
                plt.imshow(image[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
                plt.title(f"Epoch {epoch+1}")
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(Path(args.output_dir) / "samples" / f"sample_epoch_{epoch+1}.png",
                           dpi=150, bbox_inches='tight')
                plt.close()
                print(f"🎨 样本已保存")
    
    total_time = time.time() - total_start
    
    if is_main_process:
        print(f"\n✅ 训练完成！总时间: {total_time/3600:.2f} 小时")
        
        # 绘制曲线
        plt.figure(figsize=(10, 5))
        plt.plot(epoch_loss_list, label="Train")
        if val_epoch_loss_list:
            val_epochs = np.arange(args.val_interval, args.n_epochs+1, args.val_interval)
            plt.plot(val_epochs, val_epoch_loss_list, label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(Path(args.output_dir) / "learning_curves.png", dpi=150)
        plt.close()
        
        print(f"📁 输出目录: {args.output_dir}")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()

