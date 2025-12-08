# JiT TIFF Training Modification Summary

**Date**: 2024-11-28 03:52 UTC+08:00  
**Objective**: Modify JiT project to train on 1024x1024 TIFF stack images from Data directory

---

## ✅ Completed Modifications

### 1. Core Implementation

#### Created New Files:
- **`tiff_dataset.py`**: Custom PyTorch dataset classes
  - `TiffStackDataset`: Single TIFF file with multiple frames
  - `MultiTiffDataset`: Multiple TIFF files in a directory
  - Handles RGB conversion, dummy labels, and transforms

#### Modified Files:
- **`main_jit.py`**: Updated training script
  - Added TIFF dataset import
  - Changed default `img_size` from 256 to 1024
  - Changed default `data_path` to `./Data`
  - Changed default `class_num` from 1000 to 1
  - Added `--use_tiff` flag (default: True)
  - Added `--tiff_file` argument (default: mt.tif)
  - Implemented dataset selection logic

### 2. Documentation

#### Created Documentation Files:
- **`modification_plan.md`**: Detailed modification strategy
- **`modification_log.md`**: Complete change log with before/after code
- **`tasks.md`**: Task tracking and checklist
- **`README_TIFF_TRAINING.md`**: User guide for TIFF training
- **`MODIFICATION_SUMMARY.md`**: This summary document

### 3. Testing & Training Scripts

#### Created Utility Files:
- **`test_tiff_loading.py`**: Test script to verify data loading
- **`train_tiff_1024.sh`**: Linux/Mac training script
- **`train_tiff_1024.bat`**: Windows training script
- **`check_tiff.py`**: TIFF file inspection utility

---

## 🎯 Key Features

### 1. Flexible Dataset Loading
```python
# TIFF stack (default)
python main_jit.py --use_tiff --data_path ./Data --tiff_file mt.tif

# ImageNet (original)
python main_jit.py --use_tiff=False --data_path /path/to/imagenet
```

### 2. Support for 1024x1024 Images
- Default image size changed to 1024x1024
- Recommended model: JiT-B/32 (patch size 32)
- Batch size recommendations: 8-16 for 24GB GPU

### 3. Single-Class Training
- Unlabeled data support with dummy label 0
- Compatible with classifier-free guidance
- Automatic class_num detection from dataset

### 4. Backward Compatibility
- Original ImageNet training still works
- All existing parameters preserved
- Can switch between datasets with flags

---

## 📊 Configuration Recommendations

### Model Selection
| Image Size | Model | Patch Size | Patches | Memory | Recommended |
|-----------|-------|-----------|---------|--------|-------------|
| 1024x1024 | JiT-B/32 | 32x32 | 1024 | Medium | ✅ Best |
| 1024x1024 | JiT-L/32 | 32x32 | 1024 | High | ✅ More capacity |
| 1024x1024 | JiT-B/16 | 16x16 | 4096 | Very High | ⚠️ Long sequence |

### Training Parameters
```bash
--model JiT-B/32           # Recommended model
--img_size 1024            # Image size
--batch_size 16            # Adjust based on GPU memory
--blr 5e-5                 # Base learning rate
--noise_scale 2.0          # For 512/1024 images
--epochs 600               # Total epochs
--warmup_epochs 5          # Warmup period
```

---

## 🚀 Quick Start

### Step 1: Test Data Loading
```bash
python test_tiff_loading.py
```

### Step 2: Start Training
```bash
# Windows
train_tiff_1024.bat

# Linux/Mac
bash train_tiff_1024.sh
```

### Step 3: Monitor Training
- Check TensorBoard logs in `output_tiff_1024/`
- Monitor GPU memory with `nvidia-smi`
- Verify loss decreases over time

---

## 📁 File Structure

```
JiT/
├── Data/
│   └── mt.tif                      # TIFF stack (115MB)
│
├── Core Implementation
│   ├── tiff_dataset.py             # ✨ NEW: TIFF dataset classes
│   ├── main_jit.py                 # 🔧 MODIFIED: Training script
│   ├── model_jit.py                # Unchanged
│   ├── denoiser.py                 # Unchanged
│   └── engine_jit.py               # Unchanged
│
├── Testing & Training
│   ├── test_tiff_loading.py        # ✨ NEW: Test script
│   ├── train_tiff_1024.sh          # ✨ NEW: Linux training script
│   ├── train_tiff_1024.bat         # ✨ NEW: Windows training script
│   └── check_tiff.py               # ✨ NEW: TIFF inspection
│
└── Documentation
    ├── modification_plan.md         # ✨ NEW: Modification plan
    ├── modification_log.md          # ✨ NEW: Detailed change log
    ├── tasks.md                     # ✨ NEW: Task tracking
    ├── README_TIFF_TRAINING.md      # ✨ NEW: User guide
    └── MODIFICATION_SUMMARY.md      # ✨ NEW: This file
```

---

## ⚙️ Technical Details

### Dataset Implementation
- **Multi-frame support**: Loads all frames from TIFF stack
- **RGB conversion**: Automatically converts grayscale to RGB
- **Lazy loading**: Opens TIFF file per frame (memory efficient)
- **Transform pipeline**: Compatible with existing transforms
- **Dummy labels**: Returns label 0 for all frames

### Memory Considerations
- **Single image (1024x1024x3)**: ~12 MB (uint8)
- **Batch of 16**: ~192 MB
- **Model parameters (JiT-B)**: ~86M params ≈ 344 MB (fp32)
- **Activations**: Varies by model and batch size
- **Total estimate**: 8-16 GB for batch_size=16

### Training Pipeline
1. Load TIFF frames → 2. Center crop to 1024x1024 → 3. Random flip
4. Convert to tensor → 5. Normalize to [-1, 1] → 6. Forward pass
7. Compute loss → 8. Backward pass → 9. Update EMA

---

## ⚠️ Important Notes

### 1. Evaluation Disabled
- FID/IS evaluation requires reference statistics
- No reference data available for TIFF stack
- Evaluation disabled by default (no `--online_eval`)
- Can be enabled after creating reference statistics

### 2. Single Class Training
- All frames labeled as class 0
- Classifier-free guidance still functional
- May want to adjust `--label_drop_prob`

### 3. Memory Requirements
- 1024x1024 requires 16x more memory than 256x256
- Reduce batch_size if OOM occurs
- Consider gradient accumulation for small batches

### 4. Sequence Length
- Patch size 32: 1024 patches (recommended)
- Patch size 16: 4096 patches (very long)
- Longer sequences = more memory + slower training

---

## 🔍 Next Steps

### Immediate Testing
1. ✅ Run `test_tiff_loading.py` to verify data loading
2. ⏳ Run short training test (1-2 epochs)
3. ⏳ Monitor GPU memory usage
4. ⏳ Verify loss decreases

### Optimization
1. ⏳ Profile memory usage
2. ⏳ Test different batch sizes
3. ⏳ Compare model architectures (B/32 vs L/32)
4. ⏳ Adjust learning rate if needed

### Future Enhancements
- Support for 16-bit TIFF images
- Support for grayscale (single channel)
- Custom augmentations for microscopy
- Reference statistics generation
- 3D volumetric TIFF support

---

## 📚 References

- **Original Paper**: [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720)
- **Original Repo**: https://github.com/LTH14/JiT
- **Modification Date**: 2024-11-28
- **Python Version**: 3.8+
- **PyTorch Version**: 2.0+

---

## ✨ Summary

The JiT project has been successfully modified to support training on 1024x1024 TIFF stack images. The implementation:

- ✅ Maintains backward compatibility with ImageNet training
- ✅ Supports both single and multiple TIFF files
- ✅ Handles RGB conversion automatically
- ✅ Provides comprehensive documentation
- ✅ Includes testing and training scripts
- ✅ Recommends optimal configurations

**Ready to train!** Run `test_tiff_loading.py` first, then use `train_tiff_1024.bat` (Windows) or `train_tiff_1024.sh` (Linux/Mac) to start training.

---

**End of Summary**
