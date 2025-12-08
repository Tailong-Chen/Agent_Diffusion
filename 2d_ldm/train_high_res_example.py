"""
高分辨率图像生成训练示例
支持512×512和1024×1024
包含多种显存优化技巧
"""

import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# 选择配置
USE_1024 = False  # 改为True使用1024×1024配置

if USE_1024:
    from config_1024_optimized import *
    print("🚀 使用1024×1024配置")
else:
    from config_512 import *
    print("🚀 使用512×512配置")

# ============================================
# 示例：带梯度累积的训练循环
# ============================================

def train_autoencoder_with_optimization(
    autoencoderkl,
    discriminator,
    train_loader,
    val_loader,
    device,
    config
):
    """
    优化版AutoencoderKL训练循环
    包含梯度累积、梯度裁剪等技巧
    """
    
    # 损失函数
    perceptual_loss, adv_loss = get_losses(device)
    
    # 优化器
    optimizer_g = torch.optim.Adam(
        autoencoderkl.parameters(), 
        lr=config['learning_rate_g']
    )
    optimizer_d = torch.optim.Adam(
        discriminator.parameters(), 
        lr=config['learning_rate_d']
    )
    
    # 混合精度
    scaler_g = GradScaler()
    scaler_d = GradScaler()
    
    # 梯度累积步数
    accumulation_steps = OPTIMIZATION.get('gradient_accumulation_steps', 1) if USE_1024 else 1
    
    print(f"📊 训练配置:")
    print(f"   - 批次大小: {BATCH_SIZE}")
    print(f"   - 梯度累积: {accumulation_steps}")
    print(f"   - 等效批次: {BATCH_SIZE * accumulation_steps}")
    
    n_epochs = config['n_epochs']
    val_interval = config['val_interval']
    
    for epoch in range(n_epochs):
        autoencoderkl.train()
        discriminator.train()
        
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=120)
        progress_bar.set_description(f"Epoch {epoch}")
        
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            
            # ===== Generator训练 =====
            with autocast(enabled=True):
                reconstruction, z_mu, z_sigma = autoencoderkl(images)
                
                # 重建损失
                recons_loss = F.l1_loss(reconstruction.float(), images.float())
                
                # 感知损失
                p_loss = perceptual_loss(reconstruction.float(), images.float())
                
                # KL散度
                kl_loss = 0.5 * torch.sum(
                    z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, 
                    dim=[1, 2, 3]
                )
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
                
                # 总损失
                loss_g = recons_loss + \
                         (config['kl_weight'] * kl_loss) + \
                         (config['perceptual_weight'] * p_loss)
                
                # 对抗损失（预热后）
                if epoch > config['autoencoder_warm_up_n_epochs']:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += config['adv_weight'] * generator_loss
                
                # 梯度累积：除以累积步数
                loss_g = loss_g / accumulation_steps
            
            # 反向传播
            scaler_g.scale(loss_g).backward()
            
            # 每accumulation_steps步更新一次
            if (step + 1) % accumulation_steps == 0:
                # 梯度裁剪（可选）
                if USE_1024 and OPTIMIZATION.get('max_grad_norm'):
                    scaler_g.unscale_(optimizer_g)
                    torch.nn.utils.clip_grad_norm_(
                        autoencoderkl.parameters(), 
                        OPTIMIZATION['max_grad_norm']
                    )
                
                scaler_g.step(optimizer_g)
                scaler_g.update()
                optimizer_g.zero_grad(set_to_none=True)
            
            # ===== Discriminator训练 =====
            if epoch > config['autoencoder_warm_up_n_epochs']:
                with autocast(enabled=True):
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                    loss_d = config['adv_weight'] * discriminator_loss / accumulation_steps
                
                scaler_d.scale(loss_d).backward()
                
                if (step + 1) % accumulation_steps == 0:
                    scaler_d.step(optimizer_d)
                    scaler_d.update()
                    optimizer_d.zero_grad(set_to_none=True)
                
                disc_epoch_loss += discriminator_loss.item()
            
            epoch_loss += recons_loss.item()
            if epoch > config['autoencoder_warm_up_n_epochs']:
                gen_epoch_loss += generator_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                "recons": f"{epoch_loss / (step + 1):.4f}",
                "gen": f"{gen_epoch_loss / (step + 1):.4f}",
                "disc": f"{disc_epoch_loss / (step + 1):.4f}",
                "mem": f"{torch.cuda.max_memory_allocated(device)/1024**3:.1f}GB"
            })
            
            # 定期清理显存（对1024×1024有帮助）
            if USE_1024 and step % 50 == 0:
                torch.cuda.empty_cache()
        
        # 验证
        if (epoch + 1) % val_interval == 0:
            val_loss = validate(autoencoderkl, val_loader, device)
            print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")
            
            # 保存checkpoint
            save_checkpoint(autoencoderkl, discriminator, epoch, val_loss)
        
        progress_bar.close()
    
    return autoencoderkl, discriminator


def validate(model, val_loader, device):
    """验证函数"""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            with autocast(enabled=True):
                reconstruction, _, _ = model(images)
                loss = F.l1_loss(reconstruction.float(), images.float())
                val_loss += loss.item()
    
    model.train()
    return val_loss / len(val_loader)


def save_checkpoint(model, discriminator, epoch, loss):
    """保存模型checkpoint"""
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'loss': loss,
    }
    
    filename = f"{checkpoint_dir}/checkpoint_epoch_{epoch}_res_{IMAGE_SIZE}.pth"
    torch.save(checkpoint, filename)
    print(f"💾 Checkpoint保存: {filename}")


# ============================================
# 渐进式训练函数（推荐用于1024×1024）
# ============================================

def progressive_training(train_data, val_data, device):
    """
    渐进式训练：从低分辨率开始，逐步提升
    这是训练1024×1024的最佳策略
    """
    
    if not USE_1024 or not PROGRESSIVE_TRAINING['enabled']:
        print("⚠️ 渐进式训练未启用")
        return None
    
    print("🎯 开始渐进式训练...")
    
    stages = PROGRESSIVE_TRAINING['stages']
    model = None
    discriminator = None
    
    for stage_idx, stage in enumerate(stages):
        resolution = stage['resolution']
        epochs = stage['epochs']
        batch_size = stage['batch_size']
        
        print(f"\n{'='*50}")
        print(f"阶段 {stage_idx + 1}: {resolution}×{resolution}")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}")
        print(f"{'='*50}\n")
        
        # 准备该分辨率的数据加载器
        # 这里需要根据实际情况调整数据加载代码
        # train_loader = prepare_dataloader(train_data, resolution, batch_size)
        # val_loader = prepare_dataloader(val_data, resolution, batch_size)
        
        # 初始化或更新模型
        if model is None:
            # 第一次创建模型
            model = get_autoencoderkl(device)
            discriminator = get_discriminator(device)
        else:
            # 从上一阶段继续（可能需要调整某些层）
            print("📦 从上一阶段加载模型...")
        
        # 训练该阶段
        # model, discriminator = train_autoencoder_with_optimization(...)
    
    print("\n✅ 渐进式训练完成！")
    return model, discriminator


# ============================================
# 主函数示例
# ============================================

def main():
    """主训练流程"""
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  设备: {device}")
    print(f"🎨 图像分辨率: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"💾 可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 初始化模型
    autoencoderkl = get_autoencoderkl(device)
    discriminator = get_discriminator(device)
    
    print(f"\n📊 模型参数量:")
    print(f"   AutoencoderKL: {sum(p.numel() for p in autoencoderkl.parameters())/1e6:.2f}M")
    print(f"   Discriminator: {sum(p.numel() for p in discriminator.parameters())/1e6:.2f}M")
    
    # 这里需要准备实际的数据加载器
    # train_loader = ...
    # val_loader = ...
    
    print(f"\n{'='*60}")
    print("⚠️  这是一个配置和训练框架示例")
    print("   实际使用时需要:")
    print("   1. 准备对应分辨率的数据集")
    print("   2. 调整data transforms中的spatial_size")
    print("   3. 根据实际显存情况微调batch_size")
    print("   4. 监控训练过程中的显存使用")
    print(f"{'='*60}\n")
    
    # 训练
    # model, disc = train_autoencoder_with_optimization(
    #     autoencoderkl, discriminator, train_loader, val_loader,
    #     device, AUTOENCODER_CONFIG
    # )


if __name__ == "__main__":
    main()

