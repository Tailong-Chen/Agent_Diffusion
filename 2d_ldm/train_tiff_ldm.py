"""
基于MONAI的潜在扩散模型训练脚本
专门用于处理TIFF堆栈数据 (1024×1024图像)
适配32GB显存环境

使用方法:
    python train_tiff_ldm.py --tiff_path your_data.tif --output_dir ./output

作者: 基于MONAI GenerativeModels修改
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
# PyTorch AMP兼容性处理
import warnings
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*deprecated.*')
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# 图像读取库
try:
    import tifffile
    print("✅ 使用 tifffile 读取TIFF")
except ImportError:
    print("⚠️  未安装 tifffile，尝试使用 PIL")
    from PIL import Image
    tifffile = None

# MONAI 和 Generative Models
from monai import transforms
from monai.utils import set_determinism

from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler


# ============================================
# 1. TIFF堆栈数据集类
# ============================================

class TiffStackDataset(Dataset):
    """
    用于加载TIFF堆栈的数据集类
    支持单个TIFF文件包含多张图像
    """
    
    def __init__(self, tiff_path=None, transform=None, max_images=None, images_array=None):
        """
        Args:
            tiff_path: TIFF文件路径（如果images_array为None则从文件加载）
            transform: MONAI transforms
            max_images: 最多加载多少张图像（None表示全部）
            images_array: 直接提供的图像数组（避免重复加载）
        """
        self.tiff_path = tiff_path
        self.transform = transform
        
        if images_array is not None:
            # 直接使用提供的图像数组
            self.images = images_array
        else:
            # 从文件加载
            print(f"📂 加载TIFF文件: {tiff_path}")
            
            # 读取TIFF堆栈
            if tifffile is not None:
                # 使用tifffile库（推荐）
                self.images = tifffile.imread(tiff_path)
            else:
                # 使用PIL（备选方案）
                self.images = self._load_with_pil(tiff_path)
            
            # 确保是3D数组 (N, H, W)
            if self.images.ndim == 2:
                self.images = self.images[np.newaxis, ...]  # 单张图像
            elif self.images.ndim == 4:
                # 如果是 (N, H, W, C)，取第一个通道
                if self.images.shape[-1] in [1, 3, 4]:
                    self.images = self.images[..., 0]
                else:
                    self.images = self.images[:, :, :, 0]
            
            # 限制图像数量
            if max_images is not None:
                self.images = self.images[:max_images]
            
            print(f"✅ 成功加载 {self.images.shape[0]} 张图像")
            print(f"   图像形状: {self.images.shape}")
            print(f"   数据类型: {self.images.dtype}")
            print(f"   数值范围: [{self.images.min():.2f}, {self.images.max():.2f}]")
        
        self.num_images = self.images.shape[0]
    
    def _load_with_pil(self, tiff_path):
        """使用PIL加载多页TIFF"""
        images = []
        img = Image.open(tiff_path)
        
        try:
            for i in range(1000):  # 最多尝试1000页
                img.seek(i)
                images.append(np.array(img))
        except EOFError:
            pass
        
        return np.stack(images)
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        # 获取图像
        image = self.images[idx].astype(np.float32)
        
        # 添加通道维度: (H, W) -> (1, H, W)
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        
        # 转换为字典格式（MONAI风格）
        data_dict = {"image": image}
        
        # 应用transforms
        if self.transform:
            data_dict = self.transform(data_dict)
        
        return data_dict


# ============================================
# 2. 配置类
# ============================================

class LDMConfig:
    """LDM训练配置"""
    
    def __init__(self, image_size=1024, use_progressive=False):
        self.image_size = image_size
        self.use_progressive = use_progressive
        
        # 数据配置
        self.batch_size = 1 if image_size >= 1024 else 6  # 1024用batch=1以适应更大模型
        self.num_workers = 4
        self.train_split = 0.85  # 85%训练，15%验证
        
        # AutoencoderKL配置
        # 1024 -> 512 -> 256 -> 128 (3层下采样，下采样率=8)
        # 轻量级配置：减少通道数和参数量以节省显存，用长时间训练弥补
        self.autoencoder_config = {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_channels": (64, 128, 256) if image_size >= 1024 else (128, 256, 512),  # 轻量级
            "latent_channels": 3 if image_size >= 1024 else 4,  # 轻量级
            "num_res_blocks": 1,  # 轻量级
            "attention_levels": (False, False, False) if image_size >= 1024 else (False, False, True),  # 1024无attention
            "with_encoder_nonlocal_attn": False,
            "with_decoder_nonlocal_attn": False,
        }
        
        # 计算潜在空间尺寸: image_size / (2^num_layers)
        num_downsample_layers = len(self.autoencoder_config["num_channels"])
        self.latent_size = image_size // (2 ** num_downsample_layers)
        
        # Discriminator配置（轻量级）
        self.discriminator_config = {
            "spatial_dims": 2,
            "num_layers_d": 3,
            "num_channels": 32 if image_size >= 1024 else 64,  # 1024用更小的
            "in_channels": 1,
            "out_channels": 1,
        }
        
        # AutoencoderKL训练参数
        self.autoencoder_train = {
            "n_epochs": 10000 if image_size >= 1024 else 150,  # 1024用10000轮长时间训练
            "val_interval": 10,  # 验证间隔（已由用户设置）
            "warm_up_epochs": 20,  # 预热期（适中）
            "lr_g": 5e-5,
            "lr_d": 2e-4,
            "kl_weight": 1e-5 if image_size >= 1024 else 1e-6,  # 增强KL正则，使潜在空间更规整
            "perceptual_weight": 0.001,
            "adv_weight": 0.01,
        }
        
        # DiffusionModel配置（轻量级）
        latent_ch = self.autoencoder_config["latent_channels"]
        if image_size >= 1024:
            # 1024×1024增强配置（启用后两层attention提升质量）
            self.unet_config = {
                "spatial_dims": 2,
                "in_channels": latent_ch,
                "out_channels": latent_ch,
                "num_res_blocks": 2,  # 增加到2（更强学习能力）
                "num_channels": (128, 256, 512, 768),  # 增加层数和通道
                "attention_levels": (False, False, True, True),  # ✅ 启用后两层attention
                "num_head_channels": (0, 0, 512, 768),  # ✅ 匹配attention层
            }
        else:
            # 512×512标准配置
            self.unet_config = {
                "spatial_dims": 2,
                "in_channels": latent_ch,
                "out_channels": latent_ch,
                "num_res_blocks": 2,
                "num_channels": (128, 256, 512, 768),
                "attention_levels": (False, False, True, True),
                "num_head_channels": (0, 0, 512, 768),
            }
        
        # Diffusion训练参数
        self.diffusion_train = {
            "n_epochs": 10000 if image_size >= 1024 else 250,  # 1024也用10000轮
            "val_interval": 10,  # 验证间隔增加
            "lr": 1e-6 if image_size >= 1024 else 1e-4,  # ✅ 降低学习率，更稳定
            "warmup_steps": 500,  # ✅ 增加warmup步数
        }
        
        # 优化配置（针对32GB显存）
        # batch=8时约20-24GB显存，还有充足余量
        self.optimization = {
            "use_amp": True,  # 混合精度
            "gradient_accumulation_steps": 2 if image_size >= 1024 else 2,  # 减少累积，配合更大batch
            "max_grad_norm": 1.0,  # 梯度裁剪
        }
        
        # Scheduler配置
        self.scheduler_config = {
            "num_train_timesteps": 1000,
            "schedule": "scaled_linear_beta",
            "beta_start": 0.00085,
            "beta_end": 0.012,
        }
    
    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "="*60)
        print("📋 LDM训练配置")
        print("="*60)
        print(f"图像分辨率: {self.image_size}×{self.image_size}")
        print(f"潜在空间尺寸: {self.latent_size}×{self.latent_size}×{self.autoencoder_config['latent_channels']}")
        print(f"下采样率: {self.image_size // self.latent_size}x")
        print(f"批次大小: {self.batch_size}")
        print(f"梯度累积: {self.optimization['gradient_accumulation_steps']}步")
        print(f"等效批次大小: {self.batch_size * self.optimization['gradient_accumulation_steps']}")
        print(f"\nAutoEncoder训练:")
        print(f"  - Epochs: {self.autoencoder_train['n_epochs']}")
        print(f"  - 预热期: {self.autoencoder_train['warm_up_epochs']}")
        print(f"\nDiffusion训练:")
        print(f"  - Epochs: {self.diffusion_train['n_epochs']}")
        print(f"  - 推理步数: {self.scheduler_config['num_train_timesteps']}")
        print("="*60 + "\n")


# ============================================
# 3. 训练器类
# ============================================

class LDMTrainer:
    """LDM训练器"""
    
    def __init__(self, config, output_dir, device):
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "samples").mkdir(exist_ok=True)
        
        # 初始化模型
        self._init_models()
        
        # 训练历史
        self.history = {
            "autoencoder_train_loss": [],
            "autoencoder_val_loss": [],
            "diffusion_train_loss": [],
            "diffusion_val_loss": [],
        }
    
    def _init_models(self):
        """初始化所有模型"""
        print("🔧 初始化模型...")
        
        # AutoencoderKL
        self.autoencoder = AutoencoderKL(**self.config.autoencoder_config).to(self.device)
        
        # Discriminator
        self.discriminator = PatchDiscriminator(**self.config.discriminator_config).to(self.device)
        
        # UNet (稍后训练时初始化)
        self.unet = None
        
        # 损失函数
        self.perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex").to(self.device)
        self.perceptual_loss.eval()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        
        # Scheduler
        self.scheduler = DDPMScheduler(**self.config.scheduler_config)
        
        # 打印模型信息
        ae_params = sum(p.numel() for p in self.autoencoder.parameters()) / 1e6
        disc_params = sum(p.numel() for p in self.discriminator.parameters()) / 1e6
        print(f"✅ AutoencoderKL参数量: {ae_params:.2f}M")
        print(f"✅ Discriminator参数量: {disc_params:.2f}M")
    
    def train_autoencoder(self, train_loader, val_loader):
        """训练AutoencoderKL"""
        print("\n" + "="*60)
        print("🚀 开始训练 AutoencoderKL")
        print("="*60)
        
        cfg = self.config.autoencoder_train
        
        # 优化器
        optimizer_g = torch.optim.Adam(self.autoencoder.parameters(), lr=cfg["lr_g"])
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=cfg["lr_d"])
        
        # 混合精度
        scaler_g = GradScaler()
        scaler_d = GradScaler()
        
        accumulation_steps = self.config.optimization["gradient_accumulation_steps"]
        
        for epoch in range(cfg["n_epochs"]):
            self.autoencoder.train()
            self.discriminator.train()
            
            epoch_loss = 0
            gen_loss_sum = 0
            disc_loss_sum = 0
            
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120)
            progress_bar.set_description(f"Epoch {epoch+1}/{cfg['n_epochs']}")
            
            for step, batch in progress_bar:
                images = batch["image"].to(self.device)
                
                # ===== Generator训练 =====
                with autocast(enabled=self.config.optimization["use_amp"]):
                    reconstruction, z_mu, z_sigma = self.autoencoder(images)
                    
                    # 重建损失
                    recons_loss = F.l1_loss(reconstruction.float(), images.float())
                    
                    # 感知损失
                    p_loss = self.perceptual_loss(reconstruction.float(), images.float())
                    
                    # KL散度
                    kl_loss = 0.5 * torch.sum(
                        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
                        dim=[1, 2, 3]
                    )
                    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                    
                    # 总损失
                    loss_g = recons_loss + \
                             (cfg["kl_weight"] * kl_loss) + \
                             (cfg["perceptual_weight"] * p_loss)
                    
                    # 对抗损失（预热后）
                    generator_loss_val = 0
                    if epoch >= cfg["warm_up_epochs"]:
                        logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
                        generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                        loss_g += cfg["adv_weight"] * generator_loss
                        generator_loss_val = generator_loss.item()
                    
                    # 梯度累积
                    loss_g = loss_g / accumulation_steps
                
                scaler_g.scale(loss_g).backward()
                
                # 更新Generator
                if (step + 1) % accumulation_steps == 0:
                    if self.config.optimization["max_grad_norm"]:
                        scaler_g.unscale_(optimizer_g)
                        torch.nn.utils.clip_grad_norm_(
                            self.autoencoder.parameters(),
                            self.config.optimization["max_grad_norm"]
                        )
                    scaler_g.step(optimizer_g)
                    scaler_g.update()
                    optimizer_g.zero_grad(set_to_none=True)
                
                # ===== Discriminator训练 =====
                discriminator_loss_val = 0
                if epoch >= cfg["warm_up_epochs"]:
                    with autocast(enabled=self.config.optimization["use_amp"]):
                        logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
                        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                        
                        logits_real = self.discriminator(images.contiguous().detach())[-1]
                        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                        
                        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                        loss_d = cfg["adv_weight"] * discriminator_loss / accumulation_steps
                        discriminator_loss_val = discriminator_loss.item()
                    
                    scaler_d.scale(loss_d).backward()
                    
                    if (step + 1) % accumulation_steps == 0:
                        scaler_d.step(optimizer_d)
                        scaler_d.update()
                        optimizer_d.zero_grad(set_to_none=True)
                
                epoch_loss += recons_loss.item()
                gen_loss_sum += generator_loss_val
                disc_loss_sum += discriminator_loss_val
                
                # 更新进度条
                progress_bar.set_postfix({
                    "recons": f"{epoch_loss/(step+1):.4f}",
                    "gen": f"{gen_loss_sum/(step+1):.4f}",
                    "disc": f"{disc_loss_sum/(step+1):.4f}",
                    "mem": f"{torch.cuda.max_memory_allocated(self.device)/1024**3:.1f}GB"
                })
                
                # 定期清理缓存
                if step % 50 == 0:
                    torch.cuda.empty_cache()
            
            avg_train_loss = epoch_loss / len(train_loader)
            self.history["autoencoder_train_loss"].append(avg_train_loss)
            
            # 验证
            if (epoch + 1) % cfg["val_interval"] == 0:
                val_loss = self._validate_autoencoder(val_loader)
                self.history["autoencoder_val_loss"].append(val_loss)
                print(f"\n📊 Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
                
                # 保存checkpoint
                self._save_checkpoint("autoencoder", epoch, val_loss)
                
                # 保存重建样本
                self._save_reconstruction_samples(images, reconstruction, epoch, "autoencoder")
            
            progress_bar.close()
        
        print("\n✅ AutoencoderKL训练完成！")
        
        # 计算scaling factor并保存最终checkpoint
        print("📐 计算scaling factor...")
        scale_factor = self._compute_scale_factor(train_loader)
        print(f"✅ Scaling factor: {scale_factor:.4f}")
        
        # 保存包含scale_factor的最终checkpoint
        final_checkpoint = {
            "epoch": cfg["n_epochs"] - 1,
            "stage": "autoencoder",
            "loss": avg_train_loss,
            "config": self.config.__dict__,
            "autoencoder_state_dict": self.autoencoder.state_dict(),
            "scale_factor": scale_factor,
        }
        final_filename = self.output_dir / "checkpoints" / f"autoencoder_final.pth"
        torch.save(final_checkpoint, final_filename)
        print(f"💾 最终Checkpoint已保存: {final_filename}")
        
        # 清理
        del self.discriminator
        del self.perceptual_loss
        torch.cuda.empty_cache()
        
        return scale_factor
    
    def train_diffusion(self, train_loader, val_loader, scale_factor=None):
        """训练Diffusion模型"""
        print("\n" + "="*60)
        print("🚀 开始训练 Diffusion Model")
        print("="*60)
        
        # 计算scaling factor
        if scale_factor is None:
            print("📐 计算scaling factor...")
            scale_factor = self._compute_scale_factor(train_loader)
        print(f"✅ Scaling factor: {scale_factor:.4f}")
        
        # 初始化UNet
        if self.unet is None:
            self.unet = DiffusionModelUNet(**self.config.unet_config).to(self.device)
            unet_params = sum(p.numel() for p in self.unet.parameters()) / 1e6
            print(f"✅ UNet参数量: {unet_params:.2f}M")
        
        # Inferer
        inferer = LatentDiffusionInferer(self.scheduler, scale_factor=scale_factor)
        
        cfg = self.config.diffusion_train
        
        # 优化器和学习率调度
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=cfg["lr"])
        scaler = GradScaler()
        
        # 学习率warmup
        warmup_steps = cfg.get("warmup_steps", 0)
        total_steps = len(train_loader) * cfg["n_epochs"]
        
        def get_lr_multiplier(current_step):
            if warmup_steps > 0 and current_step < warmup_steps:
                return current_step / warmup_steps
            return 1.0
        
        accumulation_steps = self.config.optimization["gradient_accumulation_steps"]
        global_step = 0
        
        for epoch in range(cfg["n_epochs"]):
            self.unet.train()
            self.autoencoder.eval()
            
            epoch_loss = 0
            
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
            progress_bar.set_description(f"Epoch {epoch+1}/{cfg['n_epochs']}")
            
            for step, batch in progress_bar:
                images = batch["image"].to(self.device)
                
                # 更新学习率（warmup）
                lr_mult = get_lr_multiplier(global_step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg["lr"] * lr_mult
                
                with autocast(enabled=self.config.optimization["use_amp"]):
                    # 编码到潜在空间
                    z_mu, z_sigma = self.autoencoder.encode(images)
                    z = self.autoencoder.sampling(z_mu, z_sigma)
                    
                    # 生成噪声和timesteps
                    noise = torch.randn_like(z).to(self.device)
                    timesteps = torch.randint(
                        0, self.scheduler.num_train_timesteps,
                        (z.shape[0],), device=self.device
                    ).long()
                    
                    # 使用inferer预测噪声（与官方教程一致）
                    noise_pred = inferer(
                        inputs=images,
                        diffusion_model=self.unet,
                        noise=noise,
                        timesteps=timesteps,
                        autoencoder_model=self.autoencoder
                    )
                    
                    loss = F.mse_loss(noise_pred.float(), noise.float())
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                
                # 更新
                if (step + 1) % accumulation_steps == 0:
                    if self.config.optimization["max_grad_norm"]:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.unet.parameters(),
                            self.config.optimization["max_grad_norm"]
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                
                epoch_loss += loss.item() * accumulation_steps
                
                progress_bar.set_postfix({
                    "loss": f"{epoch_loss/(step+1):.4f}",
                    "mem": f"{torch.cuda.max_memory_allocated(self.device)/1024**3:.1f}GB"
                })
                
                if step % 50 == 0:
                    torch.cuda.empty_cache()
            
            avg_train_loss = epoch_loss / len(train_loader)
            self.history["diffusion_train_loss"].append(avg_train_loss)
            
            # 验证和采样
            if (epoch + 1) % cfg["val_interval"] == 0:
                val_loss = self._validate_diffusion(val_loader, inferer)
                self.history["diffusion_val_loss"].append(val_loss)
                print(f"\n📊 Epoch {epoch+1} - Val Loss: {val_loss:.4f}")
                
                # 保存checkpoint
                self._save_checkpoint("diffusion", epoch, val_loss, scale_factor=scale_factor)
                
                # 生成样本
                self._generate_samples(inferer, epoch, num_samples=4)
            
            progress_bar.close()
        
        print("\n✅ Diffusion Model训练完成！")
        
        return scale_factor
    
    def _validate_autoencoder(self, val_loader):
        """验证AutoencoderKL"""
        self.autoencoder.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                with autocast(enabled=True):
                    reconstruction, _, _ = self.autoencoder(images)
                    loss = F.l1_loss(reconstruction.float(), images.float())
                    val_loss += loss.item()
        
        self.autoencoder.train()
        return val_loss / len(val_loader)
    
    def _validate_diffusion(self, val_loader, inferer):
        """验证Diffusion模型"""
        self.unet.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(self.device)
                with autocast(enabled=True):
                    z_mu, z_sigma = self.autoencoder.encode(images)
                    z = self.autoencoder.sampling(z_mu, z_sigma)
                    
                    noise = torch.randn_like(z).to(self.device)
                    timesteps = torch.randint(
                        0, self.scheduler.num_train_timesteps,
                        (z.shape[0],), device=self.device
                    ).long()
                    
                    # 使用inferer预测噪声（与官方教程一致）
                    noise_pred = inferer(
                        inputs=images,
                        diffusion_model=self.unet,
                        noise=noise,
                        timesteps=timesteps,
                        autoencoder_model=self.autoencoder
                    )
                    
                    loss = F.mse_loss(noise_pred.float(), noise.float())
                    val_loss += loss.item()
        
        self.unet.train()
        return val_loss / len(val_loader)
    
    def _compute_scale_factor(self, train_loader):
        """计算scaling factor"""
        self.autoencoder.eval()
        
        with torch.no_grad():
            batch = next(iter(train_loader))
            images = batch["image"].to(self.device)
            with autocast(enabled=True):
                z = self.autoencoder.encode_stage_2_inputs(images)
        
        scale_factor = 1 / torch.std(z)
        self.autoencoder.train()
        return scale_factor.item()
    
    def _save_checkpoint(self, stage, epoch, loss, scale_factor=None):
        """保存checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "stage": stage,
            "loss": loss,
            "config": self.config.__dict__,
        }
        
        if stage == "autoencoder":
            checkpoint["autoencoder_state_dict"] = self.autoencoder.state_dict()
        elif stage == "diffusion":
            checkpoint["autoencoder_state_dict"] = self.autoencoder.state_dict()
            checkpoint["unet_state_dict"] = self.unet.state_dict()
            checkpoint["scale_factor"] = scale_factor
        
        filename = self.output_dir / "checkpoints" / f"{stage}_epoch_{epoch+1}.pth"
        torch.save(checkpoint, filename)
        print(f"💾 Checkpoint已保存: {filename}")
    
    def _save_reconstruction_samples(self, original, reconstruction, epoch, prefix):
        """保存重建样本"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        for i in range(min(4, original.shape[0])):
            # 原图
            axes[0, i].imshow(original[i, 0].cpu().detach().numpy(), cmap='gray')
            axes[0, i].set_title(f"Original {i+1}")
            axes[0, i].axis('off')
            
            # 重建
            axes[1, i].imshow(reconstruction[i, 0].cpu().detach().numpy(), cmap='gray')
            axes[1, i].set_title(f"Reconstructed {i+1}")
            axes[1, i].axis('off')
        
        plt.tight_layout()
        filename = self.output_dir / "samples" / f"{prefix}_reconstruction_epoch_{epoch+1}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"🖼️  重建样本已保存: {filename}")
    
    def _generate_samples(self, inferer, epoch, num_samples=4):
        """生成新样本（逐步去噪采样）"""
        self.unet.eval()
        self.autoencoder.eval()
        
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        if num_samples == 1:
            axes = [axes]
        
        with torch.no_grad():
            for i in range(num_samples):
                # 从标准正态分布噪声开始
                noise = torch.randn(
                    (1, self.config.autoencoder_config["latent_channels"],
                     self.config.latent_size, self.config.latent_size)
                ).to(self.device)
                
                # 设置采样步数
                self.scheduler.set_timesteps(num_inference_steps=1000)
                
                # 使用inferer.sample进行采样（与官方教程一致）
                with autocast(enabled=True):
                    sample = inferer.sample(
                        input_noise=noise,
                        diffusion_model=self.unet,
                        scheduler=self.scheduler,
                        autoencoder_model=self.autoencoder
                    )
                
                axes[i].imshow(sample[0, 0].cpu().numpy(), cmap='gray')
                axes[i].set_title(f"Sample {i+1}")
                axes[i].axis('off')
        
        plt.tight_layout()
        filename = self.output_dir / "samples" / f"generated_epoch_{epoch+1}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"🎨 生成样本已保存: {filename}")
        
        self.unet.train()
    
    def plot_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # AutoEncoder loss
        if self.history["autoencoder_train_loss"]:
            axes[0].plot(self.history["autoencoder_train_loss"], label="Train")
            if self.history["autoencoder_val_loss"]:
                val_epochs = np.linspace(
                    self.config.autoencoder_train["val_interval"],
                    len(self.history["autoencoder_train_loss"]),
                    len(self.history["autoencoder_val_loss"])
                )
                axes[0].plot(val_epochs, self.history["autoencoder_val_loss"], label="Val")
            axes[0].set_title("AutoEncoder Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True)
        
        # Diffusion loss
        if self.history["diffusion_train_loss"]:
            axes[1].plot(self.history["diffusion_train_loss"], label="Train")
            if self.history["diffusion_val_loss"]:
                val_epochs = np.linspace(
                    self.config.diffusion_train["val_interval"],
                    len(self.history["diffusion_train_loss"]),
                    len(self.history["diffusion_val_loss"])
                )
                axes[1].plot(val_epochs, self.history["diffusion_val_loss"], label="Val")
            axes[1].set_title("Diffusion Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        filename = self.output_dir / "training_history.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"📈 训练历史已保存: {filename}")


# ============================================
# 4. 主函数
# ============================================

def main():
    parser = argparse.ArgumentParser(description="训练TIFF堆栈的LDM")
    parser.add_argument("--tiff_path", type=str, required=True, help="TIFF堆栈文件路径")
    parser.add_argument("--output_dir", type=str, default="./output_ldm", help="输出目录")
    parser.add_argument("--image_size", type=int, default=1024, help="图像尺寸")
    parser.add_argument("--max_images", type=int, default=None, help="最多使用多少张图像")
    parser.add_argument("--batch_size", type=int, default=None, help="批次大小（None则自动）")
    parser.add_argument("--skip_autoencoder", action="store_true", help="跳过AutoEncoder训练")
    parser.add_argument("--skip_diffusion", action="store_true", help="跳过Diffusion训练")
    parser.add_argument("--autoencoder_checkpoint", type=str, default=None, help="AutoEncoder checkpoint路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="cuda", help="设备：cuda, cuda:0, cuda:1, cuda:2等")
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_determinism(args.seed)
    
    # 设置设备
    if "cuda" in args.device:
        if not torch.cuda.is_available():
            print("❌ 未检测到CUDA，本脚本需要GPU运行！")
            return
        device = torch.device(args.device)
        # 获取设备ID
        device_id = 0 if args.device == "cuda" else int(args.device.split(":")[-1])
        print(f"🖥️  设备: {device} (GPU {device_id})")
        print(f"💾 显存: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device(args.device)
        print(f"🖥️  设备: {device}")
    
    # 创建配置
    config = LDMConfig(image_size=args.image_size)
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    config.print_summary()
    
    # 加载数据集
    print("\n📂 加载数据集...")
    dataset = TiffStackDataset(args.tiff_path, transform=None, max_images=args.max_images)
    
    # 根据数据类型自动确定归一化范围
    sample_data = dataset.images[0]
    if sample_data.dtype == np.uint8:
        scale_max = 255.0
    elif sample_data.dtype == np.uint16:
        scale_max = 65535.0
    elif sample_data.dtype in [np.float32, np.float64]:
        scale_max = float(dataset.images.max())
    else:
        scale_max = float(dataset.images.max())
    
    print(f"📊 数据归一化范围: [0, {scale_max}] -> [0, 1]")
    
    # 数据变换 - 增强数据增强以扩充数据量
    train_transforms = transforms.Compose([
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=scale_max, 
                                        b_min=0.0, b_max=1.0, clip=True),
        transforms.Resized(keys=["image"], spatial_size=[config.image_size, config.image_size]),  # 确保尺寸正确
        transforms.RandAffined(
            keys=["image"],
            rotate_range=[(-np.pi / 12, np.pi / 12), (-np.pi / 12, np.pi / 12)],  # ✅ 增大旋转范围
            translate_range=[(-50, 50), (-50, 50)],  # ✅ 增大平移范围
            scale_range=[(-0.15, 0.15), (-0.15, 0.15)],  # ✅ 增大缩放范围
            spatial_size=[config.image_size, config.image_size],
            padding_mode="zeros",
            prob=0.9,  # ✅ 提高增强概率
        ),
        transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),  # ✅ 新增：水平翻转
        transforms.RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),  # ✅ 新增：垂直翻转
        transforms.RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.01),  # ✅ 增加概率
        transforms.RandAdjustContrastd(keys=["image"], prob=0.5, gamma=(0.7, 1.3)),  # ✅ 增强对比度范围
    ])
    
    val_transforms = transforms.Compose([
        transforms.ScaleIntensityRanged(keys=["image"], a_min=0.0, a_max=scale_max,
                                        b_min=0.0, b_max=1.0, clip=True),
        transforms.Resized(keys=["image"], spatial_size=[config.image_size, config.image_size]),
    ])
    
    # 划分训练集和验证集
    train_size = int(config.train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # 随机划分索引
    indices = list(range(len(dataset)))
    np.random.seed(args.seed)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 获取已加载的图像数组
    all_images = dataset.images
    
    # 创建独立的数据集（使用images_array参数避免重复加载TIFF）
    train_dataset = TiffStackDataset(
        images_array=all_images[train_indices],
        transform=train_transforms
    )
    val_dataset = TiffStackDataset(
        images_array=all_images[val_indices],
        transform=val_transforms
    )
    
    print(f"✅ 训练集: {train_size} 张图像")
    print(f"✅ 验证集: {val_size} 张图像")
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # 创建训练器
    trainer = LDMTrainer(config, args.output_dir, device)
    
    # 训练AutoEncoder
    scale_factor = None
    if not args.skip_autoencoder:
        if args.autoencoder_checkpoint:
            print(f"📦 加载AutoEncoder checkpoint: {args.autoencoder_checkpoint}")
            checkpoint = torch.load(args.autoencoder_checkpoint)
            trainer.autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
            scale_factor = checkpoint.get("scale_factor")
        else:
            scale_factor = trainer.train_autoencoder(train_loader, val_loader)
    else:
        print("⏭️  跳过AutoEncoder训练")
        if args.autoencoder_checkpoint:
            print(f"📦 加载AutoEncoder checkpoint: {args.autoencoder_checkpoint}")
            checkpoint = torch.load(args.autoencoder_checkpoint)
            trainer.autoencoder.load_state_dict(checkpoint["autoencoder_state_dict"])
            scale_factor = checkpoint.get("scale_factor")
    
    # 训练Diffusion
    if not args.skip_diffusion:
        trainer.train_diffusion(train_loader, val_loader, scale_factor)
    else:
        print("⏭️  跳过Diffusion训练")
    
    # 绘制训练历史
    trainer.plot_history()
    
    print("\n" + "="*60)
    print("🎉 训练完成！")
    print("="*60)
    print(f"📁 输出目录: {args.output_dir}")
    print(f"   - checkpoints/: 模型checkpoint")
    print(f"   - samples/: 生成样本")
    print(f"   - training_history.png: 训练曲线")
    print("="*60)


if __name__ == "__main__":
    main()

