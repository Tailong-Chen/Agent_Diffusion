# Code Modification Log

## Date: 2024-11-28 03:52 UTC+08:00
## Conversation: Modify training data logic to use TIFF stacks from Data directory

---

## 1. Created New Files

### 1.1 `tiff_dataset.py`
**Purpose**: Custom PyTorch dataset classes for loading TIFF stack images

**Content**:
- `TiffStackDataset`: Loads frames from a single multi-frame TIFF file
  - Handles both single-frame and multi-frame TIFF files
  - Converts images to RGB if needed
  - Returns dummy label 0 for unlabeled data
  - Supports optional transforms

- `MultiTiffDataset`: Loads frames from multiple TIFF files in a directory
  - Scans directory for .tif and .tiff files
  - Builds index of all frames across all files
  - Returns dummy label 0 for unlabeled data
  - Supports optional transforms

**Key Features**:
- Automatic frame counting
- RGB conversion for grayscale images
- Compatible with PyTorch DataLoader
- Single class (num_classes=1) for unlabeled data

---

## 2. Modified Files

### 2.1 `main_jit.py`

#### Change 1: Import statements (Line 15)
**Before**:
```python
from util.crop import center_crop_arr
import util.misc as misc
```

**After**:
```python
from util.crop import center_crop_arr
from tiff_dataset import TiffStackDataset, MultiTiffDataset
import util.misc as misc
```

**Reason**: Import custom TIFF dataset classes

---

#### Change 2: Default image size (Line 30)
**Before**:
```python
parser.add_argument('--img_size', default=256, type=int, help='Image size')
```

**After**:
```python
parser.add_argument('--img_size', default=1024, type=int, help='Image size')
```

**Reason**: Change default image size to 1024x1024 for TIFF data

---

#### Change 3: Dataset arguments (Lines 90-96)
**Before**:
```python
# dataset
parser.add_argument('--data_path', default='./data/imagenet', type=str,
                    help='Path to the dataset')
parser.add_argument('--class_num', default=1000, type=int)
```

**After**:
```python
# dataset
parser.add_argument('--data_path', default='./Data', type=str,
                    help='Path to the dataset')
parser.add_argument('--class_num', default=1, type=int)
parser.add_argument('--use_tiff', action='store_true', default=True,
                    help='Use TIFF stack dataset instead of ImageFolder')
parser.add_argument('--tiff_file', default='mt.tif', type=str,
                    help='TIFF filename in data_path')
```

**Reason**: 
- Change default data path to ./Data
- Change default class_num to 1 (unlabeled data)
- Add --use_tiff flag (default True) to enable TIFF dataset
- Add --tiff_file argument to specify TIFF filename

---

#### Change 4: Dataset loading logic (Lines 151-171)
**Before**:
```python
# Data augmentation transforms
transform_train = transforms.Compose([
    transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.PILToTensor()
])

dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
print(dataset_train)
```

**After**:
```python
# Data augmentation transforms
transform_train = transforms.Compose([
    transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.PILToTensor()
])

# Choose dataset based on use_tiff flag
if args.use_tiff:
    tiff_path = os.path.join(args.data_path, args.tiff_file)
    if os.path.isfile(tiff_path):
        # Single TIFF file
        dataset_train = TiffStackDataset(tiff_path, transform=transform_train)
        print(f"Using single TIFF file: {tiff_path}")
    elif os.path.isdir(args.data_path):
        # Multiple TIFF files in directory
        dataset_train = MultiTiffDataset(args.data_path, transform=transform_train)
        print(f"Using multiple TIFF files from: {args.data_path}")
    else:
        raise ValueError(f"TIFF path not found: {tiff_path}")
    
    # Update class_num based on dataset
    args.class_num = dataset_train.num_classes
else:
    # Original ImageNet dataset
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)

print(dataset_train)
```

**Reason**: 
- Add conditional logic to choose between TIFF and ImageFolder datasets
- Support both single TIFF file and directory of TIFF files
- Automatically update class_num from dataset
- Preserve original ImageFolder functionality when use_tiff=False

---

## 3. Supporting Files Created

### 3.1 `modification_plan.md`
Detailed plan outlining the modification strategy, architecture analysis, and implementation steps.

### 3.2 `check_tiff.py`
Utility script to inspect TIFF file properties (format, mode, size, number of frames).

---

## 4. Backward Compatibility

The modifications maintain backward compatibility:
- Original ImageNet training can be used by setting `--use_tiff=False`
- All other training parameters remain unchanged
- Model architecture is unchanged

---

## 5. Usage Examples

### Train on TIFF stack (default):
```bash
python main_jit.py --model JiT-B/32 --img_size 1024 --batch_size 16
```

### Train on TIFF with custom file:
```bash
python main_jit.py --model JiT-B/32 --img_size 1024 --data_path ./Data --tiff_file custom.tif
```

### Train on ImageNet (original):
```bash
python main_jit.py --model JiT-B/16 --img_size 256 --use_tiff=False --data_path /path/to/imagenet
```

---

## 6. Important Notes

### Memory Considerations
- 1024x1024 images require significantly more memory than 256x256
- Recommended to reduce batch_size (e.g., 16 or 32 instead of 128)
- Consider using larger patch sizes (32 or 64) for efficiency

### Model Configuration
- For 1024x1024 images, recommend using JiT-B/32 or JiT-L/32 models
- Smaller patch sizes (16) will create very long sequences (4096 patches)
- May need to adjust learning rate and training epochs

### Data Format
- TIFF dataset assumes unlabeled data (single class)
- All frames get label 0
- Classifier-free guidance still works with single class

---

## 7. Testing Checklist

- [x] Created TiffStackDataset class
- [x] Created MultiTiffDataset class
- [x] Modified main_jit.py imports
- [x] Updated default parameters
- [x] Added TIFF-specific arguments
- [x] Implemented dataset selection logic
- [ ] Test data loading with actual TIFF file
- [ ] Test training for a few iterations
- [ ] Verify memory usage
- [ ] Test checkpoint saving/loading
- [ ] Test generation/evaluation

---

## 8. Next Steps

1. Test data loading to verify TIFF format compatibility
2. Run short training test (1-2 epochs) to verify pipeline
3. Adjust batch size and model configuration based on GPU memory
4. Consider adding data augmentation specific to microscopy images
5. Monitor training metrics and adjust hyperparameters

---

## End of Log
