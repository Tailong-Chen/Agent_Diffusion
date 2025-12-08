"""
DDP多卡采样脚本
支持使用 torchrun 进行并行采样。
每个 GPU 负责生成一部分样本。
"""

import argparse
import torch
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
import os
import torch.distributed as dist
from monai.utils import set_determinism
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

def setup_ddp():
    """初始化DDP"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    else:
        # 单卡模式
        rank = 0
        world_size = 1
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    
    return rank, world_size, local_rank

def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def find_best_checkpoint(model_dir):
    """查找最佳或最新的checkpoint"""
    model_dir = Path(model_dir)
    ckpt_dir = model_dir / "checkpoints"
    
    if not ckpt_dir.exists():
        ckpt_dir = model_dir
    
    best_model = ckpt_dir / "best_model.pth"
    if best_model.exists():
        return best_model
    
    ckpts = list(ckpt_dir.glob("model_epoch_*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    
    def get_epoch(p):
        try:
            return int(p.stem.split("_")[-1])
        except:
            return 0
            
    latest_ckpt = sorted(ckpts, key=get_epoch)[-1]
    return latest_ckpt

def sample_ddpm_ddp(
    model_dir,
    output_dir,
    image_size=512,
    total_samples=100,
    batch_size=4,
    num_inference_steps=1000,
    seed=42
):
    # 1. DDP Setup
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        print(f"🚀 Starting DDP Sampling with {world_size} GPUs")
        print(f"   Total samples: {total_samples}")
        print(f"   Per GPU samples: {total_samples // world_size}")
    
    # 设置随机种子 (每个rank不同，保证生成多样性)
    set_determinism(seed + rank)
    
    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir / "generated_samples"
    else:
        output_dir = Path(output_dir)
    
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保目录已创建
    if world_size > 1:
        dist.barrier()

    # ============================================
    # 2. 初始化模型
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
    
    # 加载权重
    ckpt_path = find_best_checkpoint(model_dir)
    if rank == 0:
        print(f"   Loading checkpoint: {ckpt_path}")
        
    ckpt = torch.load(str(ckpt_path), map_location=device)
    if 'model_state_dict' in ckpt:
        unet.load_state_dict(ckpt['model_state_dict'])
    else:
        unet.load_state_dict(ckpt)
    
    unet.eval()
    
    # ============================================
    # 3. 分配任务
    # ============================================
    # 计算当前rank需要生成的数量
    samples_per_rank = total_samples // world_size
    remainder = total_samples % world_size
    if rank < remainder:
        samples_per_rank += 1
        
    # 计算全局起始索引 (用于命名)
    # 简单的做法：每个rank生成自己的，命名带rank_id，或者计算offset
    # 为了简单且不冲突，文件名格式: sample_rank{rank}_{idx}.tif
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    inferer = DiffusionInferer(scheduler)
    
    num_batches = (samples_per_rank + batch_size - 1) // batch_size
    
    if rank == 0:
        print(f"   Start sampling...")

    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, samples_per_rank - i * batch_size)
            
            noise = torch.randn((current_batch_size, 1, image_size, image_size)).to(device)
            scheduler.set_timesteps(num_inference_steps=num_inference_steps)
            
            image = noise
            
            # 只有rank 0 显示进度条，或者每个rank都显示但描述不同
            if rank == 0:
                iterator = tqdm(scheduler.timesteps, desc=f"Rank {rank} Batch {i+1}/{num_batches}")
            else:
                iterator = scheduler.timesteps
                
            for t in iterator:
                model_output = unet(
                    x=image,
                    timesteps=torch.Tensor((t,)).to(device).long()
                )
                step_result = scheduler.step(model_output, t, image)
                
                if isinstance(step_result, tuple):
                    image = step_result[0]
                else:
                    image = step_result.prev_sample
            
            # 保存图像
            for j in range(current_batch_size):
                # 唯一ID
                local_idx = i * batch_size + j
                global_idx = rank * 10000 + local_idx # 简单避免冲突
                
                img_data = image[j, 0].cpu().numpy().astype(np.float32)
                
                # 文件名: sample_r{rank}_{idx}.tif
                save_path = output_dir / f"sample_r{rank}_{local_idx:04d}.tif"
                tifffile.imwrite(save_path, img_data)
    
    if world_size > 1:
        dist.barrier()
        
    if rank == 0:
        print(f"✅ All samples generated in {output_dir}")
        
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--total_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    sample_ddpm_ddp(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        total_samples=args.total_samples,
        batch_size=args.batch_size,
        num_inference_steps=args.steps,
        seed=args.seed
    )
