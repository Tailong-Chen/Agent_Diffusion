# TIFF Stack Training Guide

This guide explains how to train JiT diffusion models on TIFF stack data.

## Overview

The codebase has been modified to support training on TIFF stack images (e.g., microscopy data) in addition to the original ImageNet dataset. The modifications allow loading multi-frame TIFF files and training on 1024x1024 images.

## Quick Start

### 1. Test Data Loading

First, verify your TIFF file can be loaded correctly:

```bash
python test_tiff_loading.py
```

This will:
- Load the TIFF file from `Data/mt.tif`
- Display frame count and image properties
- Test DataLoader functionality
- Show memory estimates for different batch sizes

### 2. Start Training

**Windows:**
```bash
train_tiff_1024.bat
```

**Linux/Mac:**
```bash
bash train_tiff_1024.sh
```

Or run directly:
```bash
python main_jit.py \
  --model JiT-B/32 \
  --img_size 1024 \
  --batch_size 16 \
  --epochs 600 \
  --data_path ./Data \
  --tiff_file mt.tif \
  --output_dir ./output_tiff_1024
```

## Configuration

### Model Selection

For 1024x1024 images, we recommend using models with larger patch sizes:

| Model | Patch Size | Patches | Parameters | Recommended |
|-------|-----------|---------|------------|-------------|
| JiT-B/16 | 16x16 | 4096 | ~86M | ⚠️ Very long sequence |
| **JiT-B/32** | **32x32** | **1024** | **~86M** | **✓ Recommended** |
| JiT-L/32 | 32x32 | 1024 | ~304M | ✓ More capacity |
| JiT-H/32 | 32x32 | 1024 | ~630M | ⚠️ High memory |

### Batch Size Recommendations

Adjust based on your GPU memory:

| GPU Memory | Recommended Batch Size | Notes |
|-----------|----------------------|-------|
| 12 GB | 4-8 | May need gradient accumulation |
| 24 GB | 8-16 | Good for most training |
| 40 GB | 16-32 | Comfortable training |
| 80 GB | 32-64 | Can use larger models |

### Key Parameters

```bash
--img_size 1024              # Image size (default: 1024)
--model JiT-B/32             # Model architecture
--batch_size 16              # Batch size per GPU
--epochs 600                 # Total training epochs
--blr 5e-5                   # Base learning rate
--noise_scale 2.0            # Noise scaling (2.0 for 512/1024)
--data_path ./Data           # Path to TIFF file directory
--tiff_file mt.tif           # TIFF filename
--use_tiff                   # Enable TIFF dataset (default: True)
--output_dir ./output        # Output directory for checkpoints
```

## Dataset Format

### Single TIFF File
Place your TIFF stack in the `Data` directory:
```
Data/
  └── mt.tif  (multi-frame TIFF)
```

### Multiple TIFF Files
Place multiple TIFF files in the `Data` directory:
```
Data/
  ├── stack1.tif
  ├── stack2.tif
  └── stack3.tif
```

The dataset will automatically:
- Load all frames from all TIFF files
- Convert grayscale to RGB if needed
- Assign dummy label 0 (unlabeled data)

## Important Notes

### 1. Memory Usage
- 1024x1024 images require ~16x more memory than 256x256
- Reduce batch size if you encounter OOM errors
- Consider using gradient accumulation for smaller batch sizes

### 2. Sequence Length
- Patch size 16: Creates 4096 patches (very long sequence)
- Patch size 32: Creates 1024 patches (manageable)
- Patch size 64: Creates 256 patches (short sequence)

### 3. Single Class Training
- TIFF dataset uses single class (class_num=1)
- All frames get label 0
- Classifier-free guidance still works

### 4. Evaluation
- FID/IS evaluation is disabled by default (no reference statistics)
- To enable, create reference statistics or modify evaluation code
- Use `--online_eval` flag only if you have reference data

## Advanced Usage

### Multi-GPU Training

```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  main_jit.py \
  --model JiT-B/32 \
  --img_size 1024 \
  --batch_size 16 \
  --epochs 600 \
  --data_path ./Data \
  --output_dir ./output_tiff_1024
```

### Resume Training

```bash
python main_jit.py \
  --model JiT-B/32 \
  --img_size 1024 \
  --resume ./output_tiff_1024 \
  --output_dir ./output_tiff_1024
```

### Custom TIFF File

```bash
python main_jit.py \
  --model JiT-B/32 \
  --img_size 1024 \
  --data_path /path/to/data \
  --tiff_file custom_stack.tif
```

### Switch Back to ImageNet

```bash
python main_jit.py \
  --model JiT-B/16 \
  --img_size 256 \
  --use_tiff=False \
  --data_path /path/to/imagenet
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` (try 8, 4, or even 2)
- Use larger patch size (JiT-B/32 instead of JiT-B/16)
- Enable gradient checkpointing (requires code modification)

### TIFF Loading Error
- Verify TIFF file path is correct
- Check TIFF format (use `test_tiff_loading.py`)
- Ensure PIL/Pillow supports your TIFF format

### Slow Training
- Increase `--num_workers` in DataLoader
- Use SSD instead of HDD for data storage
- Consider using `torch.compile()` (PyTorch 2.0+)

### Loss Not Decreasing
- Check learning rate (try adjusting `--blr`)
- Verify data loading works correctly
- Check if images are normalized properly
- Increase training epochs

## File Structure

```
JiT/
├── Data/
│   └── mt.tif                    # Your TIFF stack
├── tiff_dataset.py               # Custom TIFF dataset classes
├── main_jit.py                   # Modified training script
├── test_tiff_loading.py          # Test data loading
├── train_tiff_1024.sh            # Training script (Linux/Mac)
├── train_tiff_1024.bat           # Training script (Windows)
├── modification_plan.md          # Modification plan
├── modification_log.md           # Detailed change log
├── tasks.md                      # Task list
└── README_TIFF_TRAINING.md       # This file
```

## References

- Original JiT paper: [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720)
- Original repository: https://github.com/LTH14/JiT

## Support

For issues or questions:
1. Check the modification log: `modification_log.md`
2. Review the task list: `tasks.md`
3. Test data loading: `python test_tiff_loading.py`
4. Check GPU memory: `nvidia-smi`
