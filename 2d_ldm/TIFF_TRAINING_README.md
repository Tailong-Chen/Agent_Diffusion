# TIFF堆栈LDM训练指南

完整的训练脚本，用于从TIFF堆栈数据训练潜在扩散模型（LDM）。

---

## 📋 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [脚本说明](#脚本说明)
- [常见问题](#常见问题)

---

## 🔧 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU with CUDA support
- **显存**: 至少32GB (推荐用于1024×1024)
- **内存**: 至少32GB RAM
- **硬盘**: 根据数据集大小，建议至少50GB空闲空间

### 软件依赖

```bash
# 核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# MONAI和Generative Models
pip install monai
pip install git+https://github.com/Project-MONAI/GenerativeModels.git

# 数据处理
pip install tifffile  # 推荐
pip install pillow

# 可视化
pip install matplotlib
pip install tqdm
```

**或者使用requirements文件**:

```bash
# 创建虚拟环境
conda create -n ldm_train python=3.10
conda activate ldm_train

# 安装依赖
pip install -r requirements_tiff_ldm.txt
```

---

## 🚀 快速开始

### 1. 准备数据

确保您的TIFF堆栈文件格式正确：
- 文件格式: `.tif` 或 `.tiff`
- 图像形状: (N, H, W) 或 (N, H, W, C)
- N: 图像数量（几十张）
- H, W: 图像尺寸（1024×1024）
- 数据类型: uint8, uint16, 或 float32

**示例TIFF文件结构**:
```
your_data.tif
├── 图像1 (1024×1024)
├── 图像2 (1024×1024)
├── 图像3 (1024×1024)
└── ... (更多图像)
```

### 2. 训练模型

**完整训练** (AutoEncoder + Diffusion):

```bash
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_ldm \
    --image_size 1024 \
    --batch_size 2
```

**分步训练** (推荐，更灵活):

```bash
# 步骤1: 只训练AutoEncoder
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_ldm \
    --image_size 1024 \
    --skip_diffusion

# 步骤2: 训练Diffusion Model
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_ldm \
    --image_size 1024 \
    --skip_autoencoder \
    --autoencoder_checkpoint ./output_ldm/checkpoints/autoencoder_epoch_150.pth
```

### 3. 生成新样本

```bash
python generate_samples.py \
    --checkpoint ./output_ldm/checkpoints/diffusion_epoch_250.pth \
    --output_dir ./generated \
    --num_samples 20 \
    --num_inference_steps 1000
```

---

## 📖 详细使用

### 训练脚本参数说明

#### `train_tiff_ldm.py`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--tiff_path` | str | **必需** | TIFF堆栈文件路径 |
| `--output_dir` | str | `./output_ldm` | 输出目录 |
| `--image_size` | int | `1024` | 图像尺寸 |
| `--max_images` | int | `None` | 最多使用多少张图像 |
| `--batch_size` | int | `None` | 批次大小（自动推荐）|
| `--skip_autoencoder` | flag | `False` | 跳过AutoEncoder训练 |
| `--skip_diffusion` | flag | `False` | 跳过Diffusion训练 |
| `--autoencoder_checkpoint` | str | `None` | AutoEncoder checkpoint |
| `--seed` | int | `42` | 随机种子 |

#### 使用示例

**示例1: 使用较小的图像尺寸**

如果显存不够，可以降低分辨率：

```bash
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_512 \
    --image_size 512 \
    --batch_size 6
```

**示例2: 只使用部分数据快速测试**

```bash
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_test \
    --max_images 20 \
    --image_size 512
```

**示例3: 恢复训练**

如果训练中断，可以加载checkpoint继续：

```bash
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_ldm \
    --autoencoder_checkpoint ./output_ldm/checkpoints/autoencoder_epoch_100.pth
```

### 生成脚本参数说明

#### `generate_samples.py`

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | str | **必需** | 模型checkpoint路径 |
| `--output_dir` | str | `./generated_samples` | 输出目录 |
| `--num_samples` | int | `10` | 生成样本数量 |
| `--num_inference_steps` | int | `1000` | 推理步数（越多越好）|
| `--batch_size` | int | `1` | 批次大小 |
| `--save_intermediates` | flag | `False` | 保存去噪中间步骤 |
| `--seed` | int | `42` | 随机种子 |

#### 使用示例

**快速生成** (减少推理步数):

```bash
python generate_samples.py \
    --checkpoint ./output_ldm/checkpoints/diffusion_epoch_250.pth \
    --num_samples 10 \
    --num_inference_steps 50
```

**高质量生成** (更多推理步数):

```bash
python generate_samples.py \
    --checkpoint ./output_ldm/checkpoints/diffusion_epoch_250.pth \
    --num_samples 5 \
    --num_inference_steps 1000
```

**可视化去噪过程**:

```bash
python generate_samples.py \
    --checkpoint ./output_ldm/checkpoints/diffusion_epoch_250.pth \
    --num_samples 1 \
    --save_intermediates
```

---

## 📁 脚本说明

### 1. `train_tiff_ldm.py` - 主训练脚本

**功能**:
- 自动读取TIFF堆栈数据
- 训练AutoencoderKL（第一阶段）
- 训练Diffusion Model（第二阶段）
- 自动保存checkpoints和样本
- 绘制训练曲线

**输出结构**:
```
output_ldm/
├── checkpoints/
│   ├── autoencoder_epoch_20.pth
│   ├── autoencoder_epoch_40.pth
│   ├── ...
│   ├── diffusion_epoch_40.pth
│   └── diffusion_epoch_250.pth
├── samples/
│   ├── autoencoder_reconstruction_epoch_20.png
│   ├── generated_epoch_40.png
│   └── ...
└── training_history.png
```

**关键特性**:
- ✅ 自动数据归一化
- ✅ 自动训练/验证集划分（85%/15%）
- ✅ 混合精度训练
- ✅ 梯度累积（针对大图像）
- ✅ 梯度裁剪（防止梯度爆炸）
- ✅ 定期保存checkpoint
- ✅ 实时显存监控

### 2. `generate_samples.py` - 推理生成脚本

**功能**:
- 加载训练好的模型
- 从随机噪声生成新图像
- 保存为PNG和TIFF格式
- 创建样本网格可视化
- 可选：可视化去噪过程

**输出结构**:
```
generated_samples/
├── sample_0001.png
├── sample_0002.png
├── ...
├── sample_grid.png
├── sample_stack.tif
└── denoising_process.png (如果使用--save_intermediates)
```

### 3. 配置文件

#### `config_512.py` - 512×512配置
- 适合32GB显存
- 批次大小: 6
- 训练时间: 10-15小时

#### `config_1024_optimized.py` - 1024×1024优化配置
- 针对32GB显存优化
- 批次大小: 2
- 使用梯度累积
- 训练时间: 2-4天

---

## ⚙️ 训练配置详解

### 网络架构

#### AutoencoderKL
```python
输入: (B, 1, 1024, 1024)  # 批次大小, 通道, 高, 宽
      ↓ Encoder (3层下采样)
潜在空间: (B, 4, 128, 128)  # 下采样8倍
      ↓ Decoder (3层上采样)
输出: (B, 1, 1024, 1024)
```

**损失函数**:
- L1重建损失
- 感知损失 (AlexNet)
- KL散度损失
- 对抗损失 (PatchGAN)

#### DiffusionModelUNet
```python
输入: (B, 4, 128, 128)  # 在潜在空间操作
      ↓ UNet (含注意力机制)
输出: (B, 4, 128, 128)  # 预测噪声
```

### 训练超参数

| 阶段 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| **AutoEncoder** | Epochs | 150 | 可根据数据量调整 |
| | Learning Rate (G) | 5e-5 | Generator学习率 |
| | Learning Rate (D) | 2e-4 | Discriminator学习率 |
| | Warm-up Epochs | 15 | 预热期，不使用对抗损失 |
| | KL Weight | 1e-6 | KL散度权重 |
| | Perceptual Weight | 0.001 | 感知损失权重 |
| | Adversarial Weight | 0.01 | 对抗损失权重 |
| **Diffusion** | Epochs | 250 | 可根据数据量调整 |
| | Learning Rate | 5e-5 | UNet学习率 |
| | Timesteps | 1000 | 训练时的时间步数 |
| | Schedule | scaled_linear_beta | 噪声调度器 |

### 显存优化策略

针对32GB显存的优化（1024×1024）：

1. **混合精度训练** (FP16)
   - 节省 ~50% 显存
   - 已自动启用

2. **梯度累积** (4步)
   - 等效批次大小 = 2 × 4 = 8
   - 不增加显存消耗

3. **梯度裁剪**
   - 最大梯度范数: 1.0
   - 防止梯度爆炸

4. **定期清理缓存**
   - 每50步清理一次
   - 释放未使用的显存

---

## 📊 预期结果

### 训练时间（32GB显存）

| 分辨率 | AutoEncoder | Diffusion | 总计 |
|--------|-------------|-----------|------|
| 512×512 | ~4-5小时 | ~6-8小时 | **10-13小时** |
| 1024×1024 | ~12-18小时 | ~24-36小时 | **2-3天** |

### 显存使用

| 阶段 | 512×512 | 1024×1024 |
|------|---------|-----------|
| AutoEncoder训练 | 18-22 GB | 28-32 GB |
| Diffusion训练 | 20-24 GB | 28-30 GB |
| 推理生成 | 8-12 GB | 15-20 GB |

### Loss预期值

**AutoEncoderKL**:
- Reconstruction Loss: 应降至 < 0.02 (512) 或 < 0.03 (1024)
- Generator Loss: 应稳定在 0.2-0.3
- Discriminator Loss: 应稳定在 0.2-0.3

**Diffusion Model**:
- MSE Loss: 应收敛至 0.10-0.15
- 验证Loss: 应与训练Loss接近

---

## ❓ 常见问题

### Q1: 显存不足 (Out of Memory)

**症状**: `CUDA out of memory` 错误

**解决方案**:
```bash
# 方案1: 减小批次大小
--batch_size 1

# 方案2: 降低图像分辨率
--image_size 512

# 方案3: 使用更多梯度累积步数（脚本已自动处理）
```

### Q2: TIFF文件读取失败

**症状**: 无法加载TIFF文件

**解决方案**:
```bash
# 安装tifffile
pip install tifffile

# 如果还是失败，检查TIFF文件格式
python -c "import tifffile; print(tifffile.imread('your_data.tif').shape)"
```

### Q3: 训练速度太慢

**可能原因**:
1. 数据加载瓶颈
2. 图像尺寸太大
3. 推理步数太多

**解决方案**:
```bash
# 减少num_workers
# 在脚本中修改: config.num_workers = 2

# 使用较小的验证间隔
# 减少验证频率以加快训练

# 使用更快的采样（仅推理时）
--num_inference_steps 50  # 而不是1000
```

### Q4: 生成的图像质量不佳

**可能原因**:
1. 训练不充分
2. 数据量太少
3. 学习率不合适

**解决方案**:
1. **增加训练轮数**:
   ```bash
   # 在脚本中修改epochs配置
   ```

2. **检查数据质量**:
   - 确保TIFF图像质量良好
   - 确保有足够的数据量（至少30-50张）

3. **调整学习率**:
   - 如果Loss不下降，尝试增大学习率
   - 如果Loss震荡，尝试减小学习率

### Q5: 如何恢复中断的训练？

训练脚本会自动保存checkpoint，您可以：

```bash
# 加载最新的checkpoint继续训练
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_ldm \
    --autoencoder_checkpoint ./output_ldm/checkpoints/autoencoder_epoch_100.pth
```

### Q6: 生成的图像与训练数据风格不符

**可能原因**:
- Diffusion模型训练不充分
- Scaling factor计算不准确

**解决方案**:
1. 增加Diffusion训练轮数
2. 使用更多的推理步数生成
3. 检查AutoEncoder重建质量

---

## 📈 监控训练进度

### 1. 实时监控

训练过程中会显示：
```
Epoch 50/150: 100%|██████████| 125/125 [02:15<00:00]
recons: 0.0234  gen: 0.255  disc: 0.254  mem: 29.3GB
```

### 2. 查看训练曲线

```bash
# 查看生成的训练历史图
open output_ldm/training_history.png
```

### 3. 检查中间样本

```bash
# AutoEncoder重建样本
ls output_ldm/samples/autoencoder_reconstruction_*.png

# Diffusion生成样本
ls output_ldm/samples/generated_epoch_*.png
```

### 4. 使用TensorBoard（可选）

如果需要更详细的监控，可以修改脚本添加TensorBoard支持。

---

## 🎯 最佳实践

### 1. 数据准备
- ✅ 确保图像质量高
- ✅ 图像数量至少30-50张
- ✅ 图像内容应该相似（同一类型/风格）
- ✅ 数据预处理要一致

### 2. 训练策略
- ✅ 先用小分辨率（512）测试
- ✅ 使用分步训练（先AE后Diffusion）
- ✅ 定期检查样本质量
- ✅ 保留多个checkpoint

### 3. 生成策略
- ✅ 使用充足的推理步数（1000步）
- ✅ 生成多个样本选择最佳
- ✅ 可以调整随机种子获得不同结果

### 4. 显存管理
- ✅ 监控显存使用
- ✅ 适当调整batch_size
- ✅ 关闭不必要的进程

---

## 📚 参考资料

- [MONAI文档](https://docs.monai.io/)
- [MONAI GenerativeModels](https://github.com/Project-MONAI/GenerativeModels)
- [Latent Diffusion Models论文](https://arxiv.org/abs/2112.10752)
- [原始教程](./2d_ldm_tutorial.ipynb)

---

## 💡 提示与技巧

### 快速测试流程

```bash
# 1. 使用少量数据和小分辨率快速测试（~1小时）
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./test_run \
    --max_images 20 \
    --image_size 256 \
    --batch_size 8

# 2. 如果测试成功，再进行完整训练
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./full_run \
    --image_size 1024 \
    --batch_size 2
```

### 渐进式训练（推荐用于1024）

```bash
# 阶段1: 256×256 (快速预热)
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./progressive_256 \
    --image_size 256

# 阶段2: 512×512 (中等分辨率)
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./progressive_512 \
    --image_size 512

# 阶段3: 1024×1024 (最终分辨率)
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./progressive_1024 \
    --image_size 1024 \
    --autoencoder_checkpoint ./progressive_512/checkpoints/autoencoder_epoch_150.pth
```

---

## 🆘 获取帮助

如果遇到问题：

1. 检查本文档的"常见问题"部分
2. 查看脚本输出的错误信息
3. 检查显存使用情况
4. 确认数据格式正确

---

**祝训练顺利！🚀**

