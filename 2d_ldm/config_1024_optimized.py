# 1024×1024 配置 - 针对32GB显存的极限优化
# 使用多种优化技巧以在有限显存下训练高分辨率模型

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss

# ============================================
# 图像配置
# ============================================
IMAGE_SIZE = 1024
BATCH_SIZE = 2  # 32GB显存下的极限批次大小
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积以模拟更大batch size

# ============================================
# AutoencoderKL 配置 (1024→128或1024→64)
# ============================================
# 方案1: 3层下采样 1024 → 512 → 256 → 128 (推荐)
# 方案2: 4层下采样 1024 → 512 → 256 → 128 → 64

autoencoderkl_config = {
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1,
    "num_channels": (128, 256, 512),  # 3层: 1024→128潜在空间
    # 如果显存实在不够，可以改为 (128, 256, 512, 512) 4层: 1024→64
    "latent_channels": 4,
    "num_res_blocks": 2,  # 如果显存紧张，可以减少到1
    "attention_levels": (False, False, True),
    "with_encoder_nonlocal_attn": False,  # 关闭以节省显存
    "with_decoder_nonlocal_attn": False,
}

def get_autoencoderkl(device, use_checkpoint=True):
    """
    Args:
        use_checkpoint: 是否使用梯度检查点(gradient checkpointing)
                       可以节省50%显存但增加30%训练时间
    """
    model = AutoencoderKL(**autoencoderkl_config)
    
    # 如果需要，可以手动启用梯度检查点
    if use_checkpoint:
        # 注意：需要在forward中手动实现
        print("⚠️ 建议手动在模型forward中添加torch.utils.checkpoint.checkpoint")
    
    return model.to(device)

# ============================================
# Discriminator 配置
# ============================================
def get_discriminator(device):
    # 使用较浅的判别器以节省显存
    discriminator = PatchDiscriminator(
        spatial_dims=2, 
        num_layers_d=3,  # 不要增加层数
        num_channels=32,  # 减少通道数以节省显存
        in_channels=1, 
        out_channels=1
    )
    return discriminator.to(device)

# ============================================
# 损失函数配置
# ============================================
def get_losses(device):
    perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
    perceptual_loss.to(device)
    perceptual_loss.eval()  # 设为eval模式以节省显存
    
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    
    return perceptual_loss, adv_loss

# AutoencoderKL训练超参数
AUTOENCODER_CONFIG = {
    "kl_weight": 1e-6,
    "perceptual_weight": 0.001,
    "adv_weight": 0.01,
    "n_epochs": 200,
    "val_interval": 20,
    "autoencoder_warm_up_n_epochs": 20,  # 增加预热期
    "learning_rate_g": 5e-5,  # 稍微降低学习率
    "learning_rate_d": 2e-4,
}

# ============================================
# DiffusionModelUNet 配置
# ============================================
# 潜在空间是128×128×4
unet_config = {
    "spatial_dims": 2,
    "in_channels": 4,
    "out_channels": 4,
    "num_res_blocks": 2,
    "num_channels": (128, 256, 512, 768),  # 不要用太大的通道数
    "attention_levels": (False, False, True, True),
    "num_head_channels": (0, 0, 512, 768),
}

def get_unet(device):
    model = DiffusionModelUNet(**unet_config)
    return model.to(device)

# ============================================
# Scheduler 配置
# ============================================
def get_scheduler():
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, 
        schedule="scaled_linear_beta",  # 可以尝试不同的schedule
        beta_start=0.00085,
        beta_end=0.012
    )
    return scheduler

# 扩散模型训练超参数
DIFFUSION_CONFIG = {
    "n_epochs": 300,
    "val_interval": 50,
    "learning_rate": 5e-5,
}

# ============================================
# 优化技巧配置
# ============================================
OPTIMIZATION = {
    "use_mixed_precision": True,  # 必须使用混合精度
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "gradient_checkpointing": True,  # 梯度检查点
    "max_grad_norm": 1.0,  # 梯度裁剪
    "use_ema": False,  # EMA会占用双倍模型显存，建议关闭
}

# ============================================
# 数据加载配置
# ============================================
DATALOADER_CONFIG = {
    "batch_size": BATCH_SIZE,
    "num_workers": 4,
    "persistent_workers": True,
    "pin_memory": True,
    "prefetch_factor": 2,
}

# ============================================
# 渐进式训练配置（强烈推荐）
# ============================================
PROGRESSIVE_TRAINING = {
    "enabled": True,
    "stages": [
        {"resolution": 256, "epochs": 50, "batch_size": 16},
        {"resolution": 512, "epochs": 100, "batch_size": 6},
        {"resolution": 1024, "epochs": 200, "batch_size": 2},
    ],
    "description": "渐进式训练：从低分辨率开始，逐步提升到1024×1024"
}

print(f"""
配置总结 (1024×1024 - 32GB显存极限):
=====================================
图像分辨率: {IMAGE_SIZE}×{IMAGE_SIZE}
批次大小: {BATCH_SIZE}
梯度累积: {GRADIENT_ACCUMULATION_STEPS} (等效batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})
潜在空间: 128×128×4
下采样率: 8倍 (1024→128)
混合精度: 启用
梯度检查点: 建议启用
预计训练时间: 2-4天

⚠️ 重要提示:
1. 强烈建议使用渐进式训练
2. 必须启用混合精度训练
3. 考虑使用梯度检查点
4. 监控显存使用，必要时减小batch_size到1
5. 定期保存checkpoint以防中断
=====================================
""")

# ============================================
# 显存优化建议
# ============================================
MEMORY_TIPS = """
显存优化技巧 (按重要性排序):

1. ✅ 混合精度训练 (节省50%显存)
   - 已在原教程中使用 autocast()

2. ✅ 梯度累积 (不增加显存，模拟大batch)
   - 每N步才更新一次参数

3. ✅ 梯度检查点 (节省40-50%显存，增加30%时间)
   - 需要修改模型代码

4. ✅ 减小批次大小
   - 1024×1024下可能只能用batch_size=1或2

5. ✅ 优化器状态管理
   - 使用bitsandbytes的8-bit Adam (可选)

6. ✅ 渐进式训练
   - 从256→512→1024逐步训练

7. 清理缓存
   - 在适当时机调用 torch.cuda.empty_cache()

8. 减少验证频率
   - val_interval设大一些
"""

print(MEMORY_TIPS)

