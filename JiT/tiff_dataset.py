"""
Custom dataset for loading TIFF stack images.
Supports multi-frame TIFF files for training diffusion models.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class TiffStackDataset(Dataset):
    """
    Dataset for loading frames from a TIFF stack file.
    
    Args:
        tiff_path: Path to the TIFF file
        transform: Optional transform to be applied on a sample
        max_frames: Maximum number of frames to load (None for all frames)
    """
    
    def __init__(self, tiff_path, transform=None, max_frames=None):
        self.tiff_path = tiff_path
        self.transform = transform
        
        # Load TIFF and get number of frames
        with Image.open(tiff_path) as img:
            try:
                self.n_frames = img.n_frames
            except AttributeError:
                # Single frame TIFF
                self.n_frames = 1
            
            # Store image properties
            self.img_size = img.size
            self.img_mode = img.mode
            
            print(f"Loaded TIFF: {tiff_path}")
            print(f"  Size: {self.img_size}")
            print(f"  Mode: {self.img_mode}")
            print(f"  Frames: {self.n_frames}")
        
        # Limit frames if specified
        if max_frames is not None:
            self.n_frames = min(self.n_frames, max_frames)
        
        # For unlabeled data, use dummy label 0
        self.num_classes = 1
    
    def __len__(self):
        return self.n_frames
    
    def __getitem__(self, idx):
        """
        Load a single frame from the TIFF stack.
        
        Returns:
            image: PIL Image or transformed tensor
            label: Dummy label (0) for unlabeled data
        """
        if idx >= self.n_frames:
            raise IndexError(f"Frame index {idx} out of range [0, {self.n_frames})")
        
        # Open TIFF and seek to the requested frame
        with Image.open(self.tiff_path) as img:
            img.seek(idx)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                frame = img.convert('RGB')
            else:
                frame = img.copy()
        
        # Apply transforms if provided
        if self.transform is not None:
            frame = self.transform(frame)
        
        # Return frame with dummy label 0
        return frame, 0


class MultiTiffDataset(Dataset):
    """
    Dataset for loading frames from multiple TIFF files.
    
    Args:
        tiff_dir: Directory containing TIFF files
        transform: Optional transform to be applied on a sample
        pattern: File pattern to match (e.g., '*.tif', '*.tiff')
    """
    
    def __init__(self, tiff_dir, transform=None, pattern='*.tif'):
        self.tiff_dir = tiff_dir
        self.transform = transform
        
        # Find all TIFF files
        import glob
        tiff_files = glob.glob(os.path.join(tiff_dir, pattern))
        if not tiff_files:
            tiff_files = glob.glob(os.path.join(tiff_dir, '*.tiff'))
        
        if not tiff_files:
            raise ValueError(f"No TIFF files found in {tiff_dir}")
        
        print(f"Found {len(tiff_files)} TIFF file(s)")
        
        # Build frame index: list of (file_path, frame_idx) tuples
        self.frame_list = []
        for tiff_file in sorted(tiff_files):
            with Image.open(tiff_file) as img:
                try:
                    n_frames = img.n_frames
                except AttributeError:
                    n_frames = 1
                
                for frame_idx in range(n_frames):
                    self.frame_list.append((tiff_file, frame_idx))
                
                print(f"  {os.path.basename(tiff_file)}: {n_frames} frames")
        
        print(f"Total frames: {len(self.frame_list)}")
        self.num_classes = 1
    
    def __len__(self):
        return len(self.frame_list)
    
    def __getitem__(self, idx):
        """
        Load a single frame from the TIFF files.
        
        Returns:
            image: PIL Image or transformed tensor
            label: Dummy label (0) for unlabeled data
        """
        tiff_path, frame_idx = self.frame_list[idx]
        
        # Open TIFF and seek to the requested frame
        with Image.open(tiff_path) as img:
            if frame_idx > 0:
                img.seek(frame_idx)
            # Convert to RGB if needed
            if img.mode != 'RGB':
                frame = img.convert('RGB')
            else:
                frame = img.copy()
        
        # Apply transforms if provided
        if self.transform is not None:
            frame = self.transform(frame)
        
        # Return frame with dummy label 0
        return frame, 0
