# 16-bit TIFF 单通道数据处理指南

## 问题说明

你的 TIFF 数据是：
- **模式**: `I;16` (16-bit 单通道灰度)
- **值范围**: 0-65535
- **问题**: 直接 `convert('RGB')` 可能丢失动态范围信息

## 解决方案

### 方案 1: 使用归一化数据集（推荐）✅

**优点**:
- ✅ 正确处理 16-bit 动态范围
- ✅ 可选择逐图像或全局归一化
- ✅ 保留完整的灰度信息

**训练命令**:
```bash
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 1024 --noise_scale 2.0 \
--batch_size 18 \
--blr 5e-5 \
--epochs 60000 --warmup_epochs 5 \
--gen_bsz 8 --num_images 55 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ./output_tiff_1024_F_actin_normalized --resume ./output_tiff_1024_F_actin_normalized \
--data_path ./Data --tiff_file F-actin.tif --use_tiff \
--use_normalized_tiff \
--normalize_per_image \
--online_eval --eval_freq 50 --num_workers 4
```

**关键参数**:
- `--use_normalized_tiff`: 启用 16-bit 归一化处理
- `--normalize_per_image`: 每张图独立归一化（推荐）

### 方案 2: 保持当前方式

**优点**:
- ✅ 简单，不需要修改
- ✅ 可能已经工作

**缺点**:
- ⚠️ 可能丢失动态范围
- ⚠️ 3 倍内存开销（RGB 复制）

**当前命令**:
```bash
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32 \
--img_size 1024 \
--batch_size 18 \
--epochs 60000 \
--data_path ./Data --tiff_file F-actin.tif \
--output_dir ./output_tiff_1024_F_actin \
--online_eval --eval_freq 50
```

## 归一化方式对比

### 逐图像归一化 (Per-Image)
```python
--normalize_per_image  # 默认
```
- 每张图独立归一化到 [0, 255]
- **优点**: 每张图对比度最大化
- **缺点**: 不同图像的绝对亮度信息丢失
- **适用**: 图像间亮度差异大的情况

### 全局归一化 (Global)
```python
--normalize_per_image=False
```
- 所有图像使用相同的 min/max 归一化
- **优点**: 保留相对亮度信息
- **缺点**: 某些图像对比度可能较低
- **适用**: 需要保留绝对亮度关系

## 数据处理流程

### 原始方式
```
16-bit TIFF (0-65535)
    ↓ convert('RGB')
RGB (可能截断或缩放不当)
    ↓ PILToTensor
Tensor [0-255]
    ↓ normalize
Tensor [-1, 1]
```

### 优化方式
```
16-bit TIFF (0-65535)
    ↓ 读取为 numpy array
numpy (0-65535)
    ↓ 归一化到 0-255
numpy (0-255, uint8)
    ↓ 转为 PIL Image
PIL Image 'L' mode
    ↓ convert('RGB')
RGB (正确范围)
    ↓ PILToTensor
Tensor [0-255]
    ↓ normalize
Tensor [-1, 1]
```

## 验证数据处理

创建测试脚本验证归一化效果：

```python
from tiff_dataset_grayscale import TiffStackDatasetNormalized
import torchvision.transforms as transforms
from util.crop import center_crop_arr
import matplotlib.pyplot as plt

# 创建数据集
transform = transforms.Compose([
    transforms.Lambda(lambda img: center_crop_arr(img, 1024)),
    transforms.PILToTensor()
])

dataset = TiffStackDatasetNormalized(
    'Data/F-actin.tif',
    transform=transform,
    normalize_per_image=True
)

# 查看几张图像
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i in range(6):
    img, _ = dataset[i * 30]  # 每隔 30 帧采样
    img_np = img.numpy().transpose(1, 2, 0)
    axes[i//3, i%3].imshow(img_np)
    axes[i//3, i%3].set_title(f'Frame {i*30}')
    axes[i//3, i%3].axis('off')
plt.tight_layout()
plt.savefig('tiff_samples.png')
print(f"Saved visualization to tiff_samples.png")
```

## 推荐配置

**对于显微镜 F-actin 数据**:
```bash
--use_normalized_tiff \
--normalize_per_image
```

这样可以：
- ✅ 正确处理 16-bit 动态范围
- ✅ 每张图像对比度最大化
- ✅ 适合细胞结构可视化

## 性能对比

| 方式 | 内存 | 计算 | 数据质量 |
|------|------|------|---------|
| 原始 RGB 复制 | 3x | 3x | ⚠️ 可能丢失范围 |
| 归一化 RGB | 3x | 3x | ✅ 正确范围 |
| 单通道 (未实现) | 1x | 1x | ✅ 最优 |

## 总结

**立即使用归一化版本**:
```bash
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32 \
--img_size 1024 --batch_size 18 --epochs 60000 \
--data_path ./Data --tiff_file F-actin.tif \
--use_normalized_tiff --normalize_per_image \
--output_dir ./output_F_actin_normalized \
--online_eval --eval_freq 50
```

这样可以确保你的 16-bit 单通道数据被正确处理！
