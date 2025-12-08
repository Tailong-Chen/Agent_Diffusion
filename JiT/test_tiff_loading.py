"""
Test script to verify TIFF dataset loading works correctly.
"""

import sys
import torch
import torchvision.transforms as transforms
from tiff_dataset import TiffStackDataset, MultiTiffDataset
from util.crop import center_crop_arr

def test_tiff_dataset():
    """Test loading TIFF dataset"""
    print("=" * 60)
    print("Testing TIFF Dataset Loading")
    print("=" * 60)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, 1024)),
        transforms.PILToTensor()
    ])
    
    # Test single TIFF file
    tiff_path = 'Data/mt.tif'
    print(f"\n1. Loading TIFF file: {tiff_path}")
    print("-" * 60)
    
    try:
        dataset = TiffStackDataset(tiff_path, transform=transform)
        print(f"✓ Dataset created successfully")
        print(f"  Total frames: {len(dataset)}")
        print(f"  Number of classes: {dataset.num_classes}")
        
        # Test loading first frame
        print(f"\n2. Testing data loading...")
        print("-" * 60)
        img, label = dataset[0]
        print(f"✓ First frame loaded successfully")
        print(f"  Image shape: {img.shape}")
        print(f"  Image dtype: {img.dtype}")
        print(f"  Image min/max: {img.min():.2f} / {img.max():.2f}")
        print(f"  Label: {label}")
        
        # Test loading a few more frames
        if len(dataset) > 1:
            print(f"\n3. Testing multiple frames...")
            print("-" * 60)
            for i in range(min(3, len(dataset))):
                img, label = dataset[i]
                print(f"  Frame {i}: shape={img.shape}, label={label}")
        
        # Test with DataLoader
        print(f"\n4. Testing with DataLoader...")
        print("-" * 60)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=False,
            num_workers=0  # Use 0 for testing
        )
        
        batch_imgs, batch_labels = next(iter(dataloader))
        print(f"✓ DataLoader works successfully")
        print(f"  Batch images shape: {batch_imgs.shape}")
        print(f"  Batch labels shape: {batch_labels.shape}")
        print(f"  Batch labels: {batch_labels.tolist()}")
        
        # Memory estimate
        print(f"\n5. Memory estimates...")
        print("-" * 60)
        single_img_mb = img.numel() * img.element_size() / (1024 * 1024)
        print(f"  Single image: {single_img_mb:.2f} MB")
        batch_mb = batch_imgs.numel() * batch_imgs.element_size() / (1024 * 1024)
        print(f"  Batch of 4: {batch_mb:.2f} MB")
        print(f"  Estimated batch of 8: {batch_mb * 2:.2f} MB")
        print(f"  Estimated batch of 16: {batch_mb * 4:.2f} MB")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_tiff_dataset()
    sys.exit(0 if success else 1)
