"""
Diffusion模型生成质量诊断脚本
用于排查LDM生成效果差的问题

使用方法:
    python debug_diffusion.py --checkpoint ./output_ldm/checkpoints/diffusion_epoch_XXX.pth --tiff_path your_data.tif
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile

from monai import transforms
from torch.utils.data import DataLoader

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

# 复用train_tiff_ldm.py的数据集类
import sys
sys.path.append(str(Path(__file__).parent))
from train_tiff_ldm import TiffStackDataset


class DiffusionDebugger:
    """LDM诊断工具"""
    
    def __init__(self, checkpoint_path, tiff_path, device="cuda"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.tiff_path = tiff_path
        
        # 加载checkpoint
        print("="*60)
        print("📦 加载Checkpoint...")
        print("="*60)
        self.checkpoint = torch.load(checkpoint_path, map_location=device)
        self.config_dict = self.checkpoint["config"]
        
        # 识别checkpoint类型
        self.checkpoint_stage = self.checkpoint.get("stage", "unknown")
        print(f"📋 Checkpoint类型: {self.checkpoint_stage}")
        
        # 重建配置
        self._rebuild_config()
        
        # 初始化模型
        self._init_models()
        
        print(f"✅ Checkpoint加载成功")
        print(f"   - Epoch: {self.checkpoint['epoch'] + 1}")
        print(f"   - Loss: {self.checkpoint.get('loss', 'N/A')}")
        if self.checkpoint_stage == "diffusion":
            print(f"   - Scale Factor: {self.scale_factor:.4f}")
    
    def _rebuild_config(self):
        """从checkpoint重建配置"""
        self.image_size = self.config_dict["image_size"]
        self.autoencoder_config = self.config_dict["autoencoder_config"]
        self.latent_size = self.config_dict["latent_size"]
        
        # Diffusion相关配置（仅在diffusion checkpoint中存在）
        if self.checkpoint_stage == "diffusion":
            self.unet_config = self.config_dict["unet_config"]
            self.scheduler_config = self.config_dict["scheduler_config"]
            self.scale_factor = self.checkpoint.get("scale_factor", 1.0)
        else:
            self.unet_config = None
            self.scheduler_config = self.config_dict.get("scheduler_config")
            self.scale_factor = self.checkpoint.get("scale_factor", None)
    
    def _init_models(self):
        """初始化模型"""
        # AutoEncoder（总是需要）
        self.autoencoder = AutoencoderKL(**self.autoencoder_config).to(self.device)
        self.autoencoder.load_state_dict(self.checkpoint["autoencoder_state_dict"])
        self.autoencoder.eval()
        
        # Diffusion相关（仅在diffusion checkpoint中初始化）
        if self.checkpoint_stage == "diffusion":
            # UNet
            self.unet = DiffusionModelUNet(**self.unet_config).to(self.device)
            self.unet.load_state_dict(self.checkpoint["unet_state_dict"])
            self.unet.eval()
            
            # Scheduler
            self.scheduler = DDPMScheduler(**self.scheduler_config)
            
            # Inferer
            self.inferer = LatentDiffusionInferer(self.scheduler, scale_factor=self.scale_factor)
        else:
            self.unet = None
            self.scheduler = None
            self.inferer = None
            print("⚠️  这是AutoEncoder checkpoint，只能测试AutoEncoder功能")
    
    def load_test_data(self, num_images=10):
        """加载测试数据"""
        print("\n" + "="*60)
        print("📂 加载测试数据...")
        print("="*60)
        
        # 读取TIFF
        images = tifffile.imread(self.tiff_path)
        if images.ndim == 2:
            images = images[np.newaxis, ...]
        
        # 只取前num_images张
        images = images[:num_images]
        
        # 确定归一化范围
        if images.dtype == np.uint8:
            scale_max = 255.0
        elif images.dtype == np.uint16:
            scale_max = 65535.0
        else:
            scale_max = float(images.max())
        
        print(f"✅ 加载 {len(images)} 张图像")
        print(f"   - 数据范围: [0, {scale_max}]")
        
        # 创建数据集
        val_transforms = transforms.Compose([
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=0.0, a_max=scale_max,
                b_min=0.0, b_max=1.0, clip=True
            ),
            transforms.Resized(
                keys=["image"], 
                spatial_size=[self.image_size, self.image_size]
            ),
        ])
        
        dataset = TiffStackDataset(
            images_array=images,
            transform=val_transforms
        )
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        return loader
    
    def test_1_autoencoder_quality(self, data_loader):
        """测试1: AutoEncoder重建质量"""
        print("\n" + "="*60)
        print("🔍 测试1: AutoEncoder重建质量")
        print("="*60)
        
        recon_losses = []
        latent_stats = []
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= 5:  # 只测试前5张
                    break
                    
                images = batch["image"].to(self.device)
                
                with autocast(enabled=True):
                    # 重建
                    reconstruction, z_mu, z_sigma = self.autoencoder(images)
                    
                    # 计算损失
                    recon_loss = F.l1_loss(reconstruction.float(), images.float())
                    recon_losses.append(recon_loss.item())
                    
                    # 潜在空间统计
                    z = self.autoencoder.sampling(z_mu, z_sigma)
                    latent_stats.append({
                        "mean": z.mean().item(),
                        "std": z.std().item(),
                        "min": z.min().item(),
                        "max": z.max().item(),
                    })
        
        avg_recon_loss = np.mean(recon_losses)
        
        print(f"\n📊 重建损失 (L1): {avg_recon_loss:.6f}")
        print(f"   {'✅ 优秀' if avg_recon_loss < 0.05 else '⚠️ 较差' if avg_recon_loss < 0.1 else '❌ 很差'}")
        
        print(f"\n📊 潜在空间统计:")
        avg_stats = {k: np.mean([s[k] for s in latent_stats]) for k in latent_stats[0].keys()}
        for k, v in avg_stats.items():
            print(f"   {k}: {v:.4f}")
        
        # 判断潜在空间是否正常
        if abs(avg_stats["mean"]) > 0.5:
            print(f"   ⚠️ 警告: 潜在空间均值偏离0较多 ({avg_stats['mean']:.4f})")
        if avg_stats["std"] < 0.5 or avg_stats["std"] > 2.0:
            print(f"   ⚠️ 警告: 潜在空间标准差异常 ({avg_stats['std']:.4f}，期望接近1.0)")
        
        # 可视化重建
        self._visualize_reconstruction(data_loader)
        
        return avg_recon_loss
    
    def test_2_scaling_factor(self, data_loader):
        """测试2: Scaling Factor合理性"""
        print("\n" + "="*60)
        print("🔍 测试2: Scaling Factor")
        print("="*60)
        
        # 计算scaling factor
        with torch.no_grad():
            batch = next(iter(data_loader))
            images = batch["image"].to(self.device)
            with autocast(enabled=True):
                z = self.autoencoder.encode_stage_2_inputs(images)
        
        computed_scale = 1 / torch.std(z)
        print(f"📊 计算的Scale Factor: {computed_scale.item():.4f}")
        
        if self.scale_factor is not None:
            print(f"📊 保存的Scale Factor: {self.scale_factor:.4f}")
            diff = abs(self.scale_factor - computed_scale.item())
            if diff > 0.1:
                print(f"   ⚠️ 警告: Scale factor差异较大 (差值: {diff:.4f})")
            else:
                print(f"   ✅ Scale factor正常")
        else:
            print(f"   💡 建议使用此Scale Factor训练Diffusion模型")
        
        return computed_scale.item()
    
    def test_3_diffusion_forward(self, data_loader):
        """测试3: Diffusion前向传播"""
        if self.checkpoint_stage != "diffusion":
            print("\n⏭️  跳过测试3: 需要Diffusion checkpoint")
            return
        
        print("\n" + "="*60)
        print("🔍 测试3: Diffusion前向传播")
        print("="*60)
        
        losses_by_timestep = {t: [] for t in [0, 250, 500, 750, 999]}
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= 3:
                    break
                
                images = batch["image"].to(self.device)
                
                with autocast(enabled=True):
                    # 编码
                    z_mu, z_sigma = self.autoencoder.encode(images)
                    z = self.autoencoder.sampling(z_mu, z_sigma)
                    
                    # 测试不同timestep的预测
                    for t in losses_by_timestep.keys():
                        noise = torch.randn_like(z)
                        timesteps = torch.tensor([t], device=self.device).long()
                        
                        noise_pred = self.inferer(
                            inputs=images,
                            diffusion_model=self.unet,
                            noise=noise,
                            timesteps=timesteps,
                            autoencoder_model=self.autoencoder
                        )
                        
                        loss = F.mse_loss(noise_pred.float(), noise.float())
                        losses_by_timestep[t].append(loss.item())
        
        print(f"\n📊 不同timestep的预测损失:")
        for t, losses in losses_by_timestep.items():
            avg_loss = np.mean(losses)
            print(f"   t={t:4d}: {avg_loss:.6f}")
        
        # 检查是否学到了东西
        early_loss = np.mean(losses_by_timestep[0])
        late_loss = np.mean(losses_by_timestep[999])
        
        if early_loss > late_loss * 1.5:
            print(f"\n   ❌ 异常: 早期timestep损失过高，模型可能没学好")
        elif abs(early_loss - late_loss) < 0.01:
            print(f"\n   ⚠️ 警告: 各timestep损失几乎相同，模型可能欠拟合")
        else:
            print(f"\n   ✅ Timestep损失分布正常")
    
    def test_4_sampling_process(self, num_samples=4, num_steps=1000):
        """测试4: 采样过程"""
        if self.checkpoint_stage != "diffusion":
            print("\n⏭️  跳过测试4: 需要Diffusion checkpoint")
            return None
        
        print("\n" + "="*60)
        print(f"🔍 测试4: 采样过程 ({num_steps}步)")
        print("="*60)
        
        self.scheduler.set_timesteps(num_inference_steps=num_steps)
        
        samples = []
        intermediates_list = []
        
        with torch.no_grad():
            for i in range(num_samples):
                print(f"   生成样本 {i+1}/{num_samples}...", end=" ")
                
                # 初始噪声
                noise = torch.randn(
                    (1, self.autoencoder_config["latent_channels"],
                     self.latent_size, self.latent_size)
                ).to(self.device)
                
                # 采样
                with autocast(enabled=True):
                    sample, intermediates = self.inferer.sample(
                        input_noise=noise,
                        diffusion_model=self.unet,
                        scheduler=self.scheduler,
                        autoencoder_model=self.autoencoder,
                        save_intermediates=True,
                        intermediate_steps=num_steps // 5,  # 保存5个中间步
                    )
                
                samples.append(sample[0, 0].cpu().numpy())
                intermediates_list.append([img[0, 0].cpu().numpy() for img in intermediates])
                print("✓")
        
        # 统计生成图像
        samples_array = np.stack(samples)
        print(f"\n📊 生成图像统计:")
        print(f"   均值: {samples_array.mean():.4f}")
        print(f"   标准差: {samples_array.std():.4f}")
        print(f"   范围: [{samples_array.min():.4f}, {samples_array.max():.4f}]")
        
        # 检查异常
        if samples_array.std() < 0.05:
            print(f"   ❌ 严重问题: 生成图像几乎是常数（标准差过低）")
        elif abs(samples_array.mean() - 0.5) > 0.3:
            print(f"   ⚠️ 警告: 生成图像均值偏离0.5较多")
        else:
            print(f"   ✅ 统计特征正常")
        
        # 可视化
        self._visualize_samples(samples, intermediates_list)
        
        return samples
    
    def test_5_compare_with_real(self, data_loader, generated_samples):
        """测试5: 与真实图像对比"""
        if generated_samples is None:
            print("\n⏭️  跳过测试5: 需要Diffusion checkpoint")
            return
        
        print("\n" + "="*60)
        print("🔍 测试5: 生成图像 vs 真实图像")
        print("="*60)
        
        # 获取真实图像统计
        real_stats = []
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= 5:
                    break
                images = batch["image"].cpu().numpy()
                real_stats.append({
                    "mean": images.mean(),
                    "std": images.std(),
                    "min": images.min(),
                    "max": images.max(),
                })
        
        real_avg = {k: np.mean([s[k] for s in real_stats]) for k in real_stats[0].keys()}
        gen_array = np.stack(generated_samples)
        
        print(f"\n📊 对比:")
        print(f"   {'指标':<10} {'真实图像':<15} {'生成图像':<15} {'差异'}")
        print(f"   {'-'*50}")
        print(f"   {'均值':<10} {real_avg['mean']:<15.4f} {gen_array.mean():<15.4f} {abs(real_avg['mean'] - gen_array.mean()):.4f}")
        print(f"   {'标准差':<10} {real_avg['std']:<15.4f} {gen_array.std():<15.4f} {abs(real_avg['std'] - gen_array.std()):.4f}")
        print(f"   {'最小值':<10} {real_avg['min']:<15.4f} {gen_array.min():<15.4f} {abs(real_avg['min'] - gen_array.min()):.4f}")
        print(f"   {'最大值':<10} {real_avg['max']:<15.4f} {gen_array.max():<15.4f} {abs(real_avg['max'] - gen_array.max()):.4f}")
        
        # 判断
        mean_diff = abs(real_avg['mean'] - gen_array.mean())
        std_diff = abs(real_avg['std'] - gen_array.std())
        
        if mean_diff > 0.2 or std_diff > 0.15:
            print(f"\n   ❌ 生成图像与真实图像统计特征差异较大")
        else:
            print(f"\n   ✅ 生成图像统计特征接近真实图像")
    
    def _visualize_reconstruction(self, data_loader):
        """可视化AutoEncoder重建"""
        with torch.no_grad():
            batch = next(iter(data_loader))
            images = batch["image"].to(self.device)
            
            with autocast(enabled=True):
                reconstruction, _, _ = self.autoencoder(images)
            
            orig = images[0, 0].cpu().numpy()
            recon = reconstruction[0, 0].cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(orig, cmap='gray')
        axes[0].set_title("原始图像")
        axes[0].axis('off')
        
        axes[1].imshow(recon, cmap='gray')
        axes[1].set_title("重建图像")
        axes[1].axis('off')
        
        diff = np.abs(orig - recon)
        im = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title(f"差异图 (MAE={diff.mean():.4f})")
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        plt.tight_layout()
        output_path = Path(self.checkpoint_path).parent.parent / "debug_reconstruction.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n💾 重建对比已保存: {output_path}")
    
    def _visualize_samples(self, samples, intermediates_list):
        """可视化生成样本和去噪过程"""
        num_samples = len(samples)
        
        # 1. 最终生成样本
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        if num_samples == 1:
            axes = [axes]
        
        for i, sample in enumerate(samples):
            axes[i].imshow(sample, cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(f"样本 {i+1}")
            axes[i].axis('off')
        
        plt.tight_layout()
        output_path = Path(self.checkpoint_path).parent.parent / "debug_generated_samples.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 生成样本已保存: {output_path}")
        
        # 2. 去噪过程
        fig, axes = plt.subplots(num_samples, len(intermediates_list[0]), 
                                figsize=(len(intermediates_list[0])*3, num_samples*3))
        if num_samples == 1:
            axes = axes[np.newaxis, :]
        
        for i in range(num_samples):
            for j, intermediate in enumerate(intermediates_list[i]):
                axes[i, j].imshow(intermediate, cmap='gray', vmin=0, vmax=1)
                axes[i, j].set_title(f"样本{i+1} - 步骤{j+1}")
                axes[i, j].axis('off')
        
        plt.tight_layout()
        output_path = Path(self.checkpoint_path).parent.parent / "debug_denoising_process.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 去噪过程已保存: {output_path}")
    
    def run_full_diagnosis(self):
        """运行完整诊断"""
        print("\n" + "="*60)
        print("🔬 开始完整诊断")
        print("="*60)
        
        # 加载测试数据
        data_loader = self.load_test_data(num_images=10)
        
        # 测试1: AutoEncoder质量
        recon_loss = self.test_1_autoencoder_quality(data_loader)
        
        # 测试2: Scaling Factor
        scale_factor = self.test_2_scaling_factor(data_loader)
        
        # 测试3: Diffusion前向传播
        self.test_3_diffusion_forward(data_loader)
        
        # 测试4: 采样过程
        generated_samples = self.test_4_sampling_process(num_samples=4, num_steps=1000)
        
        # 测试5: 与真实图像对比
        self.test_5_compare_with_real(data_loader, generated_samples)
        
        # 总结
        print("\n" + "="*60)
        print("📋 诊断总结")
        print("="*60)
        
        issues = []
        suggestions = []
        
        # AutoEncoder相关问题
        if recon_loss > 0.1:
            issues.append("❌ AutoEncoder重建质量差")
            suggestions.append("   → 继续训练AutoEncoder或降低学习率")
        elif recon_loss > 0.05:
            issues.append("⚠️ AutoEncoder重建质量一般")
            suggestions.append("   → 建议继续训练以提升质量")
        
        # Scaling Factor问题
        if self.scale_factor is not None and abs(self.scale_factor - scale_factor) > 0.1:
            issues.append("⚠️ Scaling Factor不准确")
            suggestions.append("   → 重新计算并更新scaling factor")
        
        # 根据checkpoint类型给出建议
        if self.checkpoint_stage == "autoencoder":
            print("\n📌 当前状态: AutoEncoder训练完成")
            print(f"\n   AutoEncoder重建Loss: {recon_loss:.6f}")
            print(f"   建议Scaling Factor: {scale_factor:.4f}")
            
            if recon_loss < 0.05:
                print("\n✅ AutoEncoder质量优秀，可以开始训练Diffusion模型！")
                print("\n下一步:")
                print("   python train_tiff_ldm.py \\")
                print("       --tiff_path ./data/mt.tif \\")
                print("       --skip_autoencoder \\")
                print(f"       --autoencoder_checkpoint {self.checkpoint_path}")
            else:
                print("\n⚠️ 建议继续训练AutoEncoder以提升质量")
        
        elif self.checkpoint_stage == "diffusion":
            if len(issues) == 0:
                print("✅ 未发现明显问题，Diffusion模型可能需要：")
                print("   1. 继续训练更多epoch")
                print("   2. 增加训练数据量")
                print("   3. 调整学习率")
                print("   4. 尝试不同的采样步数 (如2000步)")
            else:
                print("发现以下问题:")
                for issue in issues:
                    print(f"  {issue}")
                print("\n建议:")
                for suggestion in suggestions:
                    print(f"  {suggestion}")
        
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Diffusion模型诊断工具")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Diffusion checkpoint路径")
    parser.add_argument("--tiff_path", type=str, required=True,
                       help="TIFF数据路径")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    parser.add_argument("--num_samples", type=int, default=4,
                       help="生成样本数量")
    parser.add_argument("--num_steps", type=int, default=1000,
                       help="采样步数")
    
    args = parser.parse_args()
    
    # 检查文件
    if not Path(args.checkpoint).exists():
        print(f"❌ Checkpoint不存在: {args.checkpoint}")
        return
    
    if not Path(args.tiff_path).exists():
        print(f"❌ TIFF文件不存在: {args.tiff_path}")
        return
    
    # 运行诊断
    debugger = DiffusionDebugger(args.checkpoint, args.tiff_path, args.device)
    debugger.run_full_diagnosis()


if __name__ == "__main__":
    main()

