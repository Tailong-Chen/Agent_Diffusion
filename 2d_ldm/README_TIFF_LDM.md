# TIFF堆栈潜在扩散模型（LDM）完整训练方案

基于MONAI GenerativeModels的完整训练脚本，专门用于处理TIFF堆栈数据（1024×1024图像）。

---

## 📦 文件结构

```
2d_ldm/
├── 📘 原始教程文件
│   ├── 2d_ldm_tutorial.ipynb          # 原始Jupyter教程
│   └── 2d_ldm_tutorial.py             # 原始Python教程
│
├── 🚀 TIFF训练脚本（新增）
│   ├── train_tiff_ldm.py              # ⭐ 主训练脚本
│   ├── generate_samples.py            # ⭐ 推理生成脚本
│   └── check_tiff_data.py             # ⭐ 数据检查工具
│
├── ⚙️ 配置文件
│   ├── config_512.py                  # 512×512配置
│   ├── config_1024_optimized.py       # 1024×1024优化配置
│   └── train_high_res_example.py      # 高分辨率训练示例
│
├── 📖 文档
│   ├── TIFF_TRAINING_README.md        # ⭐ 详细使用指南
│   ├── HIGH_RES_GUIDE.md              # 高分辨率训练指南
│   └── README_TIFF_LDM.md             # 本文件
│
└── 📋 依赖
    └── requirements_tiff_ldm.txt      # Python依赖列表
```

---

## 🎯 快速开始（3步）

### 1️⃣ 安装依赖

```bash
# 创建虚拟环境
conda create -n ldm_train python=3.10
conda activate ldm_train

# 安装依赖
pip install -r requirements_tiff_ldm.txt
```

### 2️⃣ 检查数据

```bash
# 检查TIFF文件是否符合要求
python check_tiff_data.py --tiff_path your_data.tif
```

### 3️⃣ 开始训练

```bash
# 完整训练（AutoEncoder + Diffusion）
python train_tiff_ldm.py \
    --tiff_path your_data.tif \
    --output_dir ./output_ldm \
    --image_size 1024 \
    --batch_size 2
```

### 4️⃣ 生成样本

```bash
# 使用训练好的模型生成新图像
python generate_samples.py \
    --checkpoint ./output_ldm/checkpoints/diffusion_epoch_250.pth \
    --num_samples 20 \
    --output_dir ./generated
```

---

## 📚 核心脚本说明

### 🔥 `train_tiff_ldm.py` - 主训练脚本

**功能**:
- 自动读取和处理TIFF堆栈
- 训练AutoencoderKL（图像压缩）
- 训练Diffusion Model（生成模型）
- 自动保存checkpoints和样本

**特点**:
- ✅ 支持1024×1024高分辨率
- ✅ 自动数据划分（训练/验证）
- ✅ 混合精度训练（节省显存）
- ✅ 梯度累积（等效更大batch）
- ✅ 实时进度显示
- ✅ 自动显存优化

**基本用法**:
```bash
python train_tiff_ldm.py --tiff_path <path> [options]
```

**重要参数**:
- `--tiff_path`: TIFF文件路径（必需）
- `--output_dir`: 输出目录
- `--image_size`: 图像尺寸（512或1024）
- `--batch_size`: 批次大小
- `--max_images`: 限制使用的图像数量

**示例**:
```bash
# 完整训练（推荐）
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_ldm \
    --image_size 1024 \
    --batch_size 2

# 只训练AutoEncoder
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_ldm \
    --skip_diffusion

# 只训练Diffusion（使用已有的AutoEncoder）
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_ldm \
    --skip_autoencoder \
    --autoencoder_checkpoint ./output_ldm/checkpoints/autoencoder_epoch_150.pth
```

---

### 🎨 `generate_samples.py` - 生成脚本

**功能**:
- 从随机噪声生成新图像
- 批量生成和保存
- 创建可视化网格
- 保存为PNG和TIFF格式

**基本用法**:
```bash
python generate_samples.py --checkpoint <path> [options]
```

**重要参数**:
- `--checkpoint`: 模型checkpoint路径（必需）
- `--num_samples`: 生成数量
- `--num_inference_steps`: 推理步数（越多质量越好）
- `--output_dir`: 输出目录
- `--save_intermediates`: 保存去噪过程

**示例**:
```bash
# 标准生成（高质量）
python generate_samples.py \
    --checkpoint ./output_ldm/checkpoints/diffusion_epoch_250.pth \
    --num_samples 20 \
    --num_inference_steps 1000

# 快速生成（测试用）
python generate_samples.py \
    --checkpoint ./output_ldm/checkpoints/diffusion_epoch_250.pth \
    --num_samples 5 \
    --num_inference_steps 50

# 可视化去噪过程
python generate_samples.py \
    --checkpoint ./output_ldm/checkpoints/diffusion_epoch_250.pth \
    --num_samples 1 \
    --save_intermediates
```

---

### 🔍 `check_tiff_data.py` - 数据检查工具

**功能**:
- 验证TIFF文件格式
- 检查图像尺寸和数量
- 分析数值范围和统计信息
- 生成可视化报告
- 提供训练建议

**基本用法**:
```bash
python check_tiff_data.py --tiff_path <path>
```

**输出**:
- 数据统计报告（终端）
- 样本图像可视化
- 统计图表

**示例**:
```bash
# 完整检查（含可视化）
python check_tiff_data.py --tiff_path ./data/your_data.tif

# 只检查不可视化
python check_tiff_data.py --tiff_path ./data/your_data.tif --no_visualize

# 指定输出目录
python check_tiff_data.py --tiff_path ./data/your_data.tif --output_dir ./check_results
```

---

## 💾 硬件要求

### 最低要求
- **GPU**: NVIDIA GPU with 16GB+ VRAM
- **RAM**: 16GB+
- **存储**: 20GB+ 空闲空间

### 推荐配置
- **GPU**: NVIDIA RTX 3090 (24GB) 或 A100 (40/80GB)
- **RAM**: 32GB+
- **存储**: 50GB+ 空闲空间

### 32GB显存下的配置

| 分辨率 | 批次大小 | 训练时间 | 状态 |
|--------|----------|----------|------|
| 512×512 | 6 | 10-15小时 | ✅ 推荐 |
| 1024×1024 | 2 | 2-4天 | ✅ 可行 |

---

## 📊 训练流程

### 完整流程图

```
数据准备
   ↓
检查数据 (check_tiff_data.py)
   ↓
训练AutoencoderKL (150 epochs, ~12-18小时)
   ├─ 学习图像压缩
   ├─ 1024×1024 → 128×128潜在空间
   └─ 保存checkpoints
   ↓
训练Diffusion Model (250 epochs, ~24-36小时)
   ├─ 学习生成潜在表示
   ├─ 在128×128空间操作
   └─ 保存checkpoints
   ↓
生成新样本 (generate_samples.py)
   ├─ 从噪声开始
   ├─ 1000步去噪
   └─ 解码到1024×1024
```

### 训练阶段详解

#### 阶段1: AutoencoderKL (图像压缩)

**目标**: 学习将1024×1024图像压缩到128×128潜在空间

**损失函数**:
- 重建损失（L1）
- 感知损失（AlexNet）
- KL散度损失
- 对抗损失（GAN）

**预期Loss**:
- 重建损失: < 0.03
- 生成器损失: 0.2-0.3
- 判别器损失: 0.2-0.3

**输出**:
- `autoencoder_epoch_*.pth`: 模型checkpoints
- `autoencoder_reconstruction_epoch_*.png`: 重建样本

#### 阶段2: Diffusion Model (生成模型)

**目标**: 学习在潜在空间中生成新的表示

**训练过程**:
1. 将图像编码到潜在空间
2. 添加随机噪声
3. 训练UNet预测噪声
4. 推理时：噪声 → 去噪 → 潜在表示 → 解码 → 图像

**预期Loss**:
- MSE损失: 0.10-0.15

**输出**:
- `diffusion_epoch_*.pth`: 模型checkpoints
- `generated_epoch_*.png`: 生成样本

---

## 📖 详细文档

- **[TIFF_TRAINING_README.md](./TIFF_TRAINING_README.md)** ⭐
  - 完整的使用指南
  - 参数详解
  - 常见问题解答
  - 最佳实践

- **[HIGH_RES_GUIDE.md](./HIGH_RES_GUIDE.md)**
  - 高分辨率训练指南
  - 显存优化技巧
  - 渐进式训练策略

---

## 🎓 使用场景

### 1. 医学图像生成
```bash
# 从几十张CT/MRI切片生成更多样本
python train_tiff_ldm.py --tiff_path medical_scans.tif
```

### 2. 显微镜图像生成
```bash
# 从显微镜图像堆栈生成新样本
python train_tiff_ldm.py --tiff_path microscopy_stack.tif
```

### 3. 材料科学图像
```bash
# 生成材料结构图像
python train_tiff_ldm.py --tiff_path material_images.tif
```

### 4. 数据增强
```bash
# 为小数据集生成增强样本
python train_tiff_ldm.py --tiff_path limited_data.tif
python generate_samples.py --checkpoint ./output/checkpoint.pth --num_samples 100
```

---

## 💡 使用技巧

### 1. 快速测试（推荐新手）

在正式训练前，先用小数据集和低分辨率测试：

```bash
# 测试运行（~1小时）
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./test_run \
    --max_images 20 \
    --image_size 256 \
    --batch_size 8
```

### 2. 渐进式训练（推荐1024×1024）

从低分辨率逐步提升：

```bash
# 第1步: 256×256（预热）
python train_tiff_ldm.py --image_size 256 --output_dir ./run_256

# 第2步: 512×512（中等）
python train_tiff_ldm.py --image_size 512 --output_dir ./run_512

# 第3步: 1024×1024（最终）
python train_tiff_ldm.py --image_size 1024 --output_dir ./run_1024 \
    --autoencoder_checkpoint ./run_512/checkpoints/autoencoder_epoch_150.pth
```

### 3. 分步训练（推荐）

分别训练两个阶段，更容易调试：

```bash
# 步骤1: 训练AutoEncoder
python train_tiff_ldm.py --skip_diffusion --output_dir ./ae_only

# 检查重建质量
# 如果满意，继续

# 步骤2: 训练Diffusion
python train_tiff_ldm.py --skip_autoencoder \
    --autoencoder_checkpoint ./ae_only/checkpoints/autoencoder_epoch_150.pth \
    --output_dir ./full_model
```

### 4. 显存不够？

```bash
# 方案1: 降低分辨率
--image_size 512

# 方案2: 减小批次大小
--batch_size 1

# 方案3: 使用更少的图像测试
--max_images 30
```

---

## 🐛 故障排除

### 问题1: CUDA Out of Memory

**症状**: 显存不足错误

**解决**:
```bash
# 减小批次大小
--batch_size 1

# 或降低分辨率
--image_size 512
```

### 问题2: TIFF加载失败

**症状**: 无法读取TIFF文件

**解决**:
```bash
# 安装tifffile
pip install tifffile

# 检查文件
python check_tiff_data.py --tiff_path your_data.tif
```

### 问题3: 训练很慢

**症状**: 每个epoch耗时很长

**解决**:
- 减少num_workers
- 使用SSD而非HDD存储数据
- 检查是否有其他进程占用GPU

### 问题4: 生成质量差

**症状**: 生成的图像模糊或有瑕疵

**解决**:
1. 增加训练轮数
2. 检查AutoEncoder重建质量
3. 使用更多推理步数（1000步）
4. 确保训练数据质量好

---

## 📈 监控训练

### 实时监控

训练过程中显示：
```
Epoch 50/150: 100%|██████| 125/125 [02:15<00:00]
recons: 0.0234  gen: 0.255  disc: 0.254  mem: 29.3GB
```

### 检查输出

```bash
# 查看训练曲线
output_ldm/training_history.png

# 查看重建样本
output_ldm/samples/autoencoder_reconstruction_epoch_*.png

# 查看生成样本
output_ldm/samples/generated_epoch_*.png
```

### 使用TensorBoard（可选）

可以修改脚本添加TensorBoard支持以获得更详细的监控。

---

## 🔗 相关资源

- **MONAI文档**: https://docs.monai.io/
- **MONAI GenerativeModels**: https://github.com/Project-MONAI/GenerativeModels
- **Latent Diffusion论文**: https://arxiv.org/abs/2112.10752
- **Stable Diffusion**: https://github.com/CompVis/stable-diffusion

---

## 📝 更新日志

### 2024版本
- ✅ 新增TIFF堆栈支持
- ✅ 优化1024×1024训练
- ✅ 添加数据检查工具
- ✅ 完善文档和示例
- ✅ 针对32GB显存优化

---

## 🙋 FAQ

**Q: 需要多少张图像？**
A: 建议至少30-50张。更多更好。

**Q: 训练需要多久？**
A: 1024×1024约2-4天，512×512约10-15小时。

**Q: 能用CPU训练吗？**
A: 理论可以，但不推荐。GPU快100倍以上。

**Q: 如何提高生成质量？**
A: 1) 增加训练轮数 2) 使用更多数据 3) 确保数据质量 4) 使用更多推理步数。

**Q: 支持彩色图像吗？**
A: 当前版本针对灰度图像。彩色图像需要修改in_channels参数。

---

## 📧 获取帮助

如果遇到问题：

1. 查看 [TIFF_TRAINING_README.md](./TIFF_TRAINING_README.md) 的FAQ部分
2. 运行 `check_tiff_data.py` 检查数据
3. 检查脚本输出的错误信息
4. 确认显存使用情况

---

## 📄 许可证

基于MONAI的Apache 2.0许可证。

---

**祝训练顺利！🚀**

有任何问题欢迎提问！

