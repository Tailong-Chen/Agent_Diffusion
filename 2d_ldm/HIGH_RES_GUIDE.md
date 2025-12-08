# 高分辨率图像生成指南 (32GB显存)

## 📋 概述

本指南说明如何使用32GB显存训练高分辨率（512×512 或 1024×1024）的潜在扩散模型。

---

## 🎯 推荐方案对比

| 特性 | 512×512 ⭐ | 1024×1024 |
|------|-----------|-----------|
| **可行性** | ✅ 强烈推荐 | ⚠️ 需要大量优化 |
| **批次大小** | 4-8 | 1-2 |
| **训练时间** | 10-15小时 | 2-4天 |
| **训练稳定性** | 高 | 中等 |
| **显存利用率** | 60-80% | 90-100% |
| **结果质量** | 优秀 | 优秀（如果训练充分） |

**结论**: 对于32GB显存，**512×512是最佳选择**。

---

## 🚀 快速开始

### 方案A: 512×512 (推荐)

#### 1. 修改原教程配置

在 `2d_ldm_tutorial.py` 中修改以下参数：

```python
# 第90行附近 - 修改图像尺寸
image_size = 512  # 原来是64

# 第108行附近 - 调整批次大小
train_loader = DataLoader(train_ds, batch_size=6, shuffle=True, num_workers=4, persistent_workers=True)

# 第132行附近 - 验证集批次大小
val_loader = DataLoader(val_ds, batch_size=6, shuffle=True, num_workers=4, persistent_workers=True)

# 第143-153行 - 修改AutoencoderKL配置
autoencoderkl = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256, 512),  # 3层下采样: 512→256→128→64
    latent_channels=4,              # 增加到4
    num_res_blocks=2,
    attention_levels=(False, False, True),  # 最高层启用注意力
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)

# 第184行附近 - 增加训练轮数
n_epochs = 150  # 原来是100

# 第303-311行 - 修改DiffusionModelUNet配置
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=4,        # 匹配latent_channels
    out_channels=4,
    num_res_blocks=2,
    num_channels=(128, 256, 512, 768),  # 增加容量
    attention_levels=(False, True, True, True),
    num_head_channels=(0, 256, 512, 768),
)

# 第344行附近 - 增加扩散模型训练轮数
n_epochs = 250  # 原来是200

# 第409行附近 - 调整潜在空间采样尺寸
# 512×512图像，3层下采样后是64×64
z = torch.randn((1, 4, 64, 64))  # 原来是(1, 3, 16, 16)

# 第449行附近 - 推理时同样调整
noise = torch.randn((1, 4, 64, 64))
```

#### 2. 修改数据变换

```python
# 第96-104行 - 训练数据变换
transforms.RandAffined(
    keys=["image"],
    rotate_range=[(-np.pi / 36, np.pi / 36), (-np.pi / 36, np.pi / 36)],
    translate_range=[(-1, 1), (-1, 1)],
    scale_range=[(-0.05, 0.05), (-0.05, 0.05)],
    spatial_size=[512, 512],  # 修改这里
    padding_mode="zeros",
    prob=0.5,
),
```

#### 3. 运行训练

```bash
# 如果使用Jupyter
jupyter notebook 2d_ldm_tutorial.ipynb

# 如果使用Python脚本
python 2d_ldm_tutorial.py
```

---

### 方案B: 1024×1024 (高级)

#### ⚠️ 重要前提条件

1. **必须使用混合精度训练** (已在原教程中使用)
2. **必须使用梯度累积**
3. **强烈建议使用渐进式训练** (256→512→1024)
4. **batch_size必须设为1或2**
5. **预计训练时间2-4天**

#### 关键配置修改

```python
# 图像尺寸
image_size = 1024

# 批次大小（极限）
batch_size = 2

# 梯度累积
gradient_accumulation_steps = 4  # 等效batch_size=8

# AutoencoderKL
autoencoderkl = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 256, 512),  # 1024→512→256→128
    latent_channels=4,
    num_res_blocks=2,
    attention_levels=(False, False, True),
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)

# UNet (潜在空间是128×128×4)
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=4,
    out_channels=4,
    num_res_blocks=2,
    num_channels=(128, 256, 512, 768),
    attention_levels=(False, False, True, True),
    num_head_channels=(0, 0, 512, 768),
)

# 训练轮数
autoencoder_epochs = 200
diffusion_epochs = 300

# 潜在空间尺寸
z = torch.randn((1, 4, 128, 128))  # 1024÷8=128
```

#### 梯度累积实现

在训练循环中添加：

```python
accumulation_steps = 4

for step, batch in progress_bar:
    images = batch["image"].to(device)
    
    with autocast(enabled=True):
        # ... 计算损失
        loss_g = loss_g / accumulation_steps  # 重要！
    
    scaler_g.scale(loss_g).backward()
    
    # 每accumulation_steps步更新一次
    if (step + 1) % accumulation_steps == 0:
        # 可选：梯度裁剪
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(autoencoderkl.parameters(), 1.0)
        
        scaler_g.step(optimizer_g)
        scaler_g.update()
        optimizer_g.zero_grad(set_to_none=True)
```

---

## 💾 显存优化技巧总结

### 已在原教程中使用的优化
- ✅ 混合精度训练 (`autocast`)
- ✅ 高效优化器设置 (`set_to_none=True`)
- ✅ 潜在扩散 (在低维空间训练)

### 针对32GB显存的额外优化

#### 1. 梯度累积 ⭐⭐⭐
```python
# 不增加显存，模拟更大的batch size
accumulation_steps = 4
loss = loss / accumulation_steps
```

#### 2. 降低批次大小 ⭐⭐⭐
```python
# 1024×1024时
batch_size = 2  # 甚至可能需要1
```

#### 3. 减少验证频率 ⭐⭐
```python
val_interval = 50  # 增大间隔
```

#### 4. 定期清理缓存 ⭐
```python
if step % 50 == 0:
    torch.cuda.empty_cache()
```

#### 5. 梯度检查点 ⭐⭐⭐ (高级)
```python
from torch.utils.checkpoint import checkpoint

# 需要修改模型forward函数
# 可节省40-50%显存，但增加30%训练时间
```

#### 6. 8-bit优化器 ⭐⭐ (可选)
```python
# 安装: pip install bitsandbytes
import bitsandbytes as bnb
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=1e-4)
```

---

## 📊 预期显存使用

### 512×512配置
- **AutoencoderKL训练**: ~18-22 GB
- **DiffusionModel训练**: ~20-24 GB
- **推理**: ~8-12 GB

### 1024×1024配置
- **AutoencoderKL训练**: ~28-32 GB (满载!)
- **DiffusionModel训练**: ~28-30 GB
- **推理**: ~15-20 GB

**监控命令**:
```bash
# 实时监控显存
watch -n 1 nvidia-smi

# Python中监控
print(f"当前显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"峰值显存: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
```

---

## 🎓 渐进式训练策略 (强烈推荐用于1024×1024)

### 为什么使用渐进式训练？
1. 更稳定的训练过程
2. 更快的收敛速度
3. 更好的最终效果
4. 更少的显存压力

### 三阶段训练方案

#### 阶段1: 256×256 (基础阶段)
```python
image_size = 256
batch_size = 16
epochs = 50
# 训练AutoencoderKL + Diffusion
```

#### 阶段2: 512×512 (过渡阶段)
```python
image_size = 512
batch_size = 6
epochs = 100
# 加载阶段1的权重，继续训练
```

#### 阶段3: 1024×1024 (最终阶段)
```python
image_size = 1024
batch_size = 2
epochs = 200
# 加载阶段2的权重，最终fine-tune
```

### 权重迁移
```python
# 从低分辨率加载权重
checkpoint = torch.load('checkpoint_512.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 继续训练更高分辨率
# 注意：潜在空间尺寸会改变，UNet需要适配
```

---

## 🐛 常见问题

### Q1: 训练时显存溢出 (OOM)
**解决方案**:
1. 减小batch_size到2或1
2. 启用梯度累积
3. 减少num_res_blocks到1
4. 考虑使用更多下采样层
5. 降低num_channels

### Q2: 训练速度太慢
**解决方案**:
1. 减少num_workers
2. 使用更少的验证步骤
3. 不使用梯度检查点
4. 考虑从512×512开始

### Q3: 生成的图像质量不佳
**解决方案**:
1. 增加训练轮数
2. 检查scaling_factor是否合适
3. 调整学习率
4. 使用渐进式训练
5. 确保数据质量良好

### Q4: 如何恢复训练？
```python
# 保存checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# 加载checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 📈 训练监控建议

### 关键指标

1. **AutoencoderKL阶段**:
   - 重建损失 (recons_loss): 应持续下降
   - 目标: < 0.02 (512×512), < 0.03 (1024×1024)

2. **Diffusion阶段**:
   - MSE损失: 应收敛到0.10-0.15
   - 每40个epoch检查生成质量

3. **显存使用**:
   - 512×512: 不应超过28GB
   - 1024×1024: 应保持在31GB以下

### 可视化建议
```python
# 定期保存生成样本
if epoch % 10 == 0:
    with torch.no_grad():
        sample = inferer.sample(...)
        plt.imsave(f'sample_epoch_{epoch}.png', sample[0,0].cpu())
```

---

## ✅ 推荐工作流程

### 对于512×512:
1. ✅ 直接使用本指南的配置
2. ✅ 训练150 epochs AutoencoderKL (~4小时)
3. ✅ 训练250 epochs DiffusionModel (~6小时)
4. ✅ 总计约10-12小时即可获得优质结果

### 对于1024×1024:
1. ✅ 强烈建议使用渐进式训练
2. ✅ 或使用预训练的512×512模型fine-tune
3. ✅ 密切监控显存使用
4. ✅ 使用梯度累积和其他优化技巧
5. ✅ 准备2-4天的训练时间

---

## 📚 参考资料

- [Stable Diffusion论文](https://arxiv.org/abs/2112.10752)
- [MONAI Generative Models文档](https://github.com/Project-MONAI/GenerativeModels)
- 原教程: `2d_ldm_tutorial.ipynb`
- 配置文件: `config_512.py`, `config_1024_optimized.py`
- 训练示例: `train_high_res_example.py`

---

## 💡 最终建议

**对于您的32GB显存环境:**

1. **首选512×512**: 最佳的效果/成本平衡
2. **数据质量**: 确保训练数据质量高且充足
3. **监控训练**: 实时监控loss和生成质量
4. **保存checkpoint**: 每10-20 epochs保存一次
5. **耐心等待**: 高质量生成需要充分训练

**预期效果**:
- 512×512: 10-15小时获得优质结果 ⭐⭐⭐⭐⭐
- 1024×1024: 2-4天可能获得优质结果 ⭐⭐⭐

祝训练顺利！🚀

