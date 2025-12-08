# 🚀 快速开始指南

5分钟上手TIFF堆栈LDM训练！

---

## 📦 准备工作（一次性）

### 1. 安装依赖

```bash
# 方法1: 使用pip（推荐）
pip install -r requirements_tiff_ldm.txt

# 方法2: 分步安装
pip install torch torchvision torchaudio
pip install monai
pip install tifffile matplotlib tqdm
```

### 2. 准备数据

确保您有一个TIFF堆栈文件：
- 格式: `.tif` 或 `.tiff`
- 包含: 几十张1024×1024的灰度图像
- 位置: 例如 `./data/your_data.tif`

---

## ⚡ 方式1: 使用脚本（最简单）

### Windows用户

1. 编辑 `quick_start.bat`，修改第13行：
   ```batch
   set "TIFF_PATH=.\data\your_data.tif"
   ```

2. 双击运行 `quick_start.bat`

### Linux/Mac用户

1. 编辑 `quick_start.sh`，修改第14行：
   ```bash
   TIFF_PATH="./data/your_data.tif"
   ```

2. 运行：
   ```bash
   chmod +x quick_start.sh
   ./quick_start.sh
   ```

---

## ⚡ 方式2: 命令行（推荐）

### 步骤1: 检查数据 ✅

```bash
python check_tiff_data.py --tiff_path ./data/your_data.tif
```

**输出示例**:
```
✅ 成功加载 45 张图像
   图像形状: (45, 1024, 1024)
✅ 图像数量充足 (45张)
✅ 标准1024×1024尺寸
💡 推荐命令（1024×1024）:
   python train_tiff_ldm.py --tiff_path ...
```

### 步骤2: 开始训练 🚀

**选项A: 完整训练（推荐新手）**

```bash
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_ldm \
    --image_size 1024 \
    --batch_size 2
```

**选项B: 分步训练（推荐高级用户）**

```bash
# 第1步: 训练AutoEncoder（12-18小时）
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_ldm \
    --image_size 1024 \
    --skip_diffusion

# 第2步: 训练Diffusion（24-36小时）
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./output_ldm \
    --image_size 1024 \
    --skip_autoencoder \
    --autoencoder_checkpoint ./output_ldm/checkpoints/autoencoder_epoch_150.pth
```

### 步骤3: 生成样本 🎨

```bash
python generate_samples.py \
    --checkpoint ./output_ldm/checkpoints/diffusion_epoch_250.pth \
    --num_samples 20 \
    --output_dir ./generated
```

---

## 💡 根据显存调整

### 32GB显存（您的配置）

**1024×1024（原始质量）**:
```bash
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --image_size 1024 \
    --batch_size 2
```

**512×512（更快，推荐测试）**:
```bash
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --image_size 512 \
    --batch_size 6
```

### 16GB显存

```bash
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --image_size 512 \
    --batch_size 2
```

### 显存不够？

```bash
# 最小配置
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --image_size 256 \
    --batch_size 4
```

---

## 📊 训练监控

### 实时进度

训练时会显示：
```
Epoch 50/150: 100%|██████| 125/125 [02:15<00:00]
recons: 0.0234  gen: 0.255  disc: 0.254  mem: 29.3GB
```

### 查看输出

```bash
# 训练曲线
output_ldm/training_history.png

# 重建样本
output_ldm/samples/autoencoder_reconstruction_epoch_*.png

# 生成样本
output_ldm/samples/generated_epoch_*.png

# Checkpoints
output_ldm/checkpoints/*.pth
```

---

## ⏱️ 预期时间（32GB显存）

| 分辨率 | AutoEncoder | Diffusion | 总计 |
|--------|-------------|-----------|------|
| 512×512 | 4-5小时 | 6-8小时 | **10-13小时** |
| 1024×1024 | 12-18小时 | 24-36小时 | **2-3天** |

---

## 🆘 常见问题

### ❌ CUDA Out of Memory

```bash
# 减小批次大小
--batch_size 1

# 或降低分辨率
--image_size 512
```

### ❌ TIFF加载失败

```bash
# 安装tifffile
pip install tifffile

# 检查文件
python check_tiff_data.py --tiff_path your_data.tif
```

### ❌ 找不到模块

```bash
# 重新安装依赖
pip install -r requirements_tiff_ldm.txt

# 检查MONAI GenerativeModels
pip install git+https://github.com/Project-MONAI/GenerativeModels.git
```

### ❓ 生成质量不好

1. **增加训练轮数**: 在配置中调整epochs
2. **使用更多推理步数**: `--num_inference_steps 1000`
3. **检查AutoEncoder**: 确保重建质量好
4. **增加数据量**: 至少30-50张图像

---

## 📚 详细文档

- **完整指南**: [TIFF_TRAINING_README.md](./TIFF_TRAINING_README.md)
- **高分辨率指南**: [HIGH_RES_GUIDE.md](./HIGH_RES_GUIDE.md)
- **项目总览**: [README_TIFF_LDM.md](./README_TIFF_LDM.md)

---

## 🎯 快速测试（推荐新手）

在正式训练前先快速测试（~1小时）：

```bash
python train_tiff_ldm.py \
    --tiff_path ./data/your_data.tif \
    --output_dir ./test_run \
    --max_images 20 \
    --image_size 256 \
    --batch_size 8
```

如果测试成功，再进行完整训练！

---

## 📋 完整流程总结

```bash
# 1. 检查数据（1分钟）
python check_tiff_data.py --tiff_path your_data.tif

# 2. 快速测试（可选，1小时）
python train_tiff_ldm.py --max_images 20 --image_size 256

# 3. 完整训练（2-3天）
python train_tiff_ldm.py --image_size 1024

# 4. 生成样本（5-10分钟）
python generate_samples.py --checkpoint output_ldm/checkpoints/diffusion_epoch_250.pth
```

---

## 💡 最后的提示

✅ **推荐**: 先用512×512训练（10-15小时），确认效果后再考虑1024×1024

✅ **推荐**: 使用分步训练，可以随时检查和调整

✅ **推荐**: 定期检查生成的样本质量

✅ **推荐**: 保留多个checkpoint，选择最佳的

---

**准备好了吗？开始训练吧！🚀**

如有问题，请查看详细文档：[TIFF_TRAINING_README.md](./TIFF_TRAINING_README.md)

