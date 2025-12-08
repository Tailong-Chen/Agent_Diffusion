# 512×512 配置 - 针对32GB显存优化
# 基于原教程修改，适合生成高分辨率医学图像

import torch
import torch.nn.functional as F
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler
from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss

# ============================================
# 图像配置
# ============================================
IMAGE_SIZE = 512
BATCH_SIZE = 6  # 32GB显存下的安全批次大小

# ============================================
# AutoencoderKL 配置 (512→64潜在空间)
# ============================================
# 下采样路径: 512 → 256 → 128 → 64
autoencoderkl_config = {
    "spatial_dims": 2,
    "in_channels": 1,
    "out_channels": 1,
    "num_channels": (128, 256, 512),  # 3层下采样
    "latent_channels": 4,  # 增加潜在通道以保留更多信息
    "num_res_blocks": 2,
    "attention_levels": (False, False, True),  # 最高层使用注意力
    "with_encoder_nonlocal_attn": False,
    "with_decoder_nonlocal_attn": False,
}

def get_autoencoderkl(device):
    model = AutoencoderKL(**autoencoderkl_config)
    return model.to(device)

# ============================================
# Discriminator 配置
# ============================================
def get_discriminator(device):
    discriminator = PatchDiscriminator(
        spatial_dims=2, 
        num_layers_d=3,  # 可以增加到4层以处理更大图像
        num_channels=64, 
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
    
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    
    return perceptual_loss, adv_loss

# AutoencoderKL训练超参数
AUTOENCODER_CONFIG = {
    "kl_weight": 1e-6,
    "perceptual_weight": 0.001,
    "adv_weight": 0.01,
    "n_epochs": 150,  # 更大图像可能需要更多epoch
    "val_interval": 10,
    "autoencoder_warm_up_n_epochs": 10,
    "learning_rate_g": 1e-4,
    "learning_rate_d": 5e-4,
}

# ============================================
# DiffusionModelUNet 配置
# ============================================
# 潜在空间是64×64×4
unet_config = {
    "spatial_dims": 2,
    "in_channels": 4,  # 匹配latent_channels
    "out_channels": 4,
    "num_res_blocks": 2,
    "num_channels": (128, 256, 512, 768),  # 增加容量
    "attention_levels": (False, True, True, True),  # 后三层使用注意力
    "num_head_channels": (0, 256, 512, 768),
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
        schedule="linear_beta", 
        beta_start=0.0015, 
        beta_end=0.0195
    )
    return scheduler

# 扩散模型训练超参数
DIFFUSION_CONFIG = {
    "n_epochs": 250,  # 更大图像需要更多epoch
    "val_interval": 40,
    "learning_rate": 1e-4,
}

# ============================================
# 推理配置
# ============================================
INFERENCE_CONFIG = {
    "num_inference_steps": 1000,
    "guidance_scale": 1.0,  # 如果使用条件生成
}

# ============================================
# 数据加载配置
# ============================================
DATALOADER_CONFIG = {
    "batch_size": BATCH_SIZE,
    "num_workers": 4,
    "persistent_workers": True,
    "pin_memory": True,  # 加速数据传输
}

print(f"""
配置总结 (32GB显存优化):
=====================================
图像分辨率: {IMAGE_SIZE}×{IMAGE_SIZE}
批次大小: {BATCH_SIZE}
潜在空间: 64×64×4
下采样率: 8倍 ({IMAGE_SIZE}→64)
AutoEncoder epochs: {AUTOENCODER_CONFIG['n_epochs']}
Diffusion epochs: {DIFFUSION_CONFIG['n_epochs']}
预计训练时间: 10-15小时
=====================================
""")

