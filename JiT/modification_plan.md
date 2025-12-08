# Modification Plan for TIFF Stack Training

## Date: 2024-11-28 03:52 UTC+08:00
## Conversation: Understanding project and modifying training data logic to use TIFF stacks

## Objective
Modify the JiT project to train on TIFF stack data from the Data directory instead of ImageNet dataset.
- Image size: 1024x1024
- Data format: TIFF stack (mt.tif)

## Current Architecture Analysis

### 1. Data Loading (main_jit.py)
- Currently uses `torchvision.datasets.ImageFolder` for ImageNet
- Expects directory structure: data_path/train/class_folders/images
- Uses `center_crop_arr` for preprocessing
- Applies random horizontal flip augmentation

### 2. Model Architecture (model_jit.py)
- JiT model with various sizes (B/16, B/32, L/16, L/32, H/16, H/32)
- Patch-based processing with configurable patch sizes
- Supports variable input sizes (currently 256x256 or 512x512)
- Uses class labels for conditional generation

### 3. Training Logic (engine_jit.py)
- Normalizes images to [-1, 1]
- Uses class labels for conditional training
- Implements label dropout for classifier-free guidance

## Required Modifications

### 1. Create Custom TIFF Dataset Class
- Load TIFF stack from Data/mt.tif
- Extract individual frames from the stack
- Handle 1024x1024 image size
- Provide dummy labels (since this is likely unlabeled microscopy data)

### 2. Modify main_jit.py
- Replace ImageFolder dataset with custom TiffStackDataset
- Update default image size to 1024
- Adjust class_num parameter (set to 1 for single-class or unlabeled data)
- Update data augmentation pipeline if needed

### 3. Update Model Configuration
- Ensure model can handle 1024x1024 images
- May need to use larger patch size (32 or 64) for computational efficiency
- Adjust positional embeddings for larger image size

### 4. Considerations
- TIFF stacks may contain 3D volumetric data - need to handle as 2D slices
- Memory requirements will be higher for 1024x1024 images
- May need to reduce batch size
- Consider if this is grayscale or RGB data

## Implementation Steps

1. Create `tiff_dataset.py` with custom dataset class
2. Modify `main_jit.py` to use the new dataset
3. Update default parameters for 1024x1024 training
4. Test data loading before full training
5. Document changes in modification log

## Files to Modify
- main_jit.py (data loading logic)
- New file: tiff_dataset.py (custom dataset)
- Potentially: util/crop.py (if special preprocessing needed)

## Files to Create
- tiff_dataset.py: Custom dataset for TIFF stacks
- modification_log.md: Detailed record of code changes
