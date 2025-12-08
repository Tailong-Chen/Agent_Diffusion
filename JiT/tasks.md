# Task List

## Date: 2024-11-28 03:52 UTC+08:00
## Conversation: Modify JiT to train on TIFF stacks

---

## Completed Tasks ✓

- [x] Analyze project structure and understand codebase
- [x] Create modification plan document
- [x] Create custom TiffStackDataset class
- [x] Create MultiTiffDataset class for multiple TIFF files
- [x] Modify main_jit.py to import TIFF dataset classes
- [x] Update default image size to 1024x1024
- [x] Update default data path to ./Data
- [x] Update default class_num to 1
- [x] Add --use_tiff argument
- [x] Add --tiff_file argument
- [x] Implement dataset selection logic in main_jit.py
- [x] Create detailed modification log
- [x] Create this task list

---

## Pending Tasks (High Priority)

### Testing & Validation
- [ ] Test TIFF data loading
  - Verify mt.tif can be loaded correctly
  - Check frame count and dimensions
  - Verify RGB conversion works
  - Test with DataLoader

- [ ] Test training pipeline
  - Run 1-2 training iterations
  - Verify forward pass works with 1024x1024 images
  - Check memory usage
  - Verify loss computation

- [ ] Adjust hyperparameters
  - Reduce batch_size if needed (recommend 8-16 for 1024x1024)
  - Consider using JiT-B/32 or JiT-L/32 for larger patch size
  - Adjust learning rate if needed

---

## Pending Tasks (Medium Priority)

### Model Configuration
- [ ] Test different model architectures
  - JiT-B/32 (recommended for 1024x1024)
  - JiT-L/32 (if more capacity needed)
  - Compare memory usage and training speed

- [ ] Optimize for 1024x1024 images
  - Verify positional embeddings work correctly
  - Check if rope embeddings need adjustment
  - Monitor attention computation time

### Training Configuration
- [x] Create training script for TIFF data
  - Set appropriate batch_size
  - Set appropriate learning rate
  - Set appropriate number of epochs
  - Configure checkpoint saving

- [x] Setup evaluation
  - Disable or modify FID/IS evaluation (no reference stats for TIFF data)
  - Add custom evaluation metrics if needed
  - Configure generation parameters

---

## Pending Tasks (Low Priority)

### Documentation
- [x] Update README.md with TIFF training instructions
- [x] Document recommended hyperparameters for 1024x1024
- [x] Add example training commands

### Data Augmentation
- [ ] Consider microscopy-specific augmentations
  - Rotation (90, 180, 270 degrees)
  - Brightness/contrast adjustments
  - Gaussian noise augmentation
  - Elastic deformations

### Optimization
- [ ] Profile memory usage
- [ ] Consider gradient checkpointing for larger models
- [ ] Test mixed precision training effectiveness
- [ ] Consider using torch.compile for speedup

### Future Enhancements
- [ ] Support for grayscale TIFF (single channel)
- [ ] Support for 16-bit TIFF images
- [ ] Support for 3D volumetric TIFF stacks
- [ ] Add data normalization options
- [ ] Add frame sampling strategies (e.g., skip frames)

---

## Known Issues / Considerations

1. **Memory Usage**: 1024x1024 images require ~16x more memory than 256x256
   - Solution: Reduce batch_size significantly (8-16 recommended)
   - Consider using gradient accumulation

2. **Patch Count**: With patch_size=16, 1024x1024 creates 4096 patches
   - This is a very long sequence for transformer
   - Recommendation: Use patch_size=32 (1024 patches) or 64 (256 patches)

3. **Single Class**: Dataset uses dummy label 0 for all frames
   - Classifier-free guidance still works
   - May want to disable label dropout or adjust probability

4. **No Reference Statistics**: Cannot compute FID/IS without reference data
   - Need to disable online_eval or create custom metrics
   - Could generate reference statistics from training data

5. **TIFF Format Compatibility**: Need to verify mt.tif format
   - Check if it's RGB or grayscale
   - Check if it's 8-bit or 16-bit
   - Verify number of frames

---

## Immediate Next Steps

1. **Test Data Loading** (CRITICAL)
   ```bash
   python -c "from tiff_dataset import TiffStackDataset; import torchvision.transforms as T; from util.crop import center_crop_arr; ds = TiffStackDataset('Data/mt.tif', transform=T.Compose([T.Lambda(lambda x: center_crop_arr(x, 1024)), T.PILToTensor()])); print(f'Dataset size: {len(ds)}'); img, label = ds[0]; print(f'Image shape: {img.shape}, Label: {label}')"
   ```

2. **Run Quick Training Test**
   ```bash
   python main_jit.py --model JiT-B/32 --img_size 1024 --batch_size 8 --epochs 1 --output_dir ./test_output
   ```

3. **Monitor and Adjust**
   - Check GPU memory usage
   - Verify training loss decreases
   - Adjust batch_size if OOM errors occur

---

## End of Task List
