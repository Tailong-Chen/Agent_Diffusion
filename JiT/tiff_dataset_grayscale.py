"""
Optimized dataset for single-channel (grayscale) TIFF stacks.
More memory and compute efficient for microscopy data.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class TiffStackDatasetGrayscale(Dataset):
    """
    Dataset for loading single-channel frames from a TIFF stack.
    Optimized for grayscale microscopy images.
    
    Args:
        tiff_path: Path to the TIFF file
        transform: Optional transform to be applied on a sample
        max_frames: Maximum number of frames to load (None for all frames)
        keep_single_channel: If True, keep as 1-channel; if False, replicate to 3-channel RGB
    """
    
    def __init__(self, tiff_path, transform=None, max_frames=None, keep_single_channel=False):
        self.tiff_path = tiff_path
        self.transform = transform
        self.keep_single_channel = keep_single_channel
        
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
            print(f"  Single channel mode: {keep_single_channel}")
        
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
            image: PIL Image or transformed tensor (1-channel or 3-channel)
            label: Dummy label (0) for unlabeled data
        """
        if idx >= self.n_frames:
            raise IndexError(f"Frame index {idx} out of range [0, {self.n_frames})")
        
        # Open TIFF and seek to the requested frame
        with Image.open(self.tiff_path) as img:
            img.seek(idx)
            
            if self.keep_single_channel:
                # Keep as grayscale (more efficient)
                if img.mode in ['I;16', 'I;16B', 'I;16L', 'I;16N']:
                    # 16-bit image, normalize to 8-bit for consistency
                    frame_array = np.array(img)
                    # Normalize to 0-255 range
                    frame_array = ((frame_array - frame_array.min()) / 
                                   (frame_array.max() - frame_array.min() + 1e-8) * 255).astype(np.uint8)
                    frame = Image.fromarray(frame_array, mode='L')
                elif img.mode != 'L':
                    frame = img.convert('L')
                else:
                    frame = img.copy()
            else:
                # Convert to RGB (original behavior)
                if img.mode != 'RGB':
                    frame = img.convert('RGB')
                else:
                    frame = img.copy()
        
        # Apply transforms if provided
        if self.transform is not None:
            frame = self.transform(frame)
        
        # If keeping single channel and frame is tensor, ensure it has channel dimension
        if self.keep_single_channel and isinstance(frame, torch.Tensor):
            if frame.ndim == 2:  # [H, W]
                frame = frame.unsqueeze(0)  # [1, H, W]
        
        # Return frame with dummy label 0
        return frame, 0


class TiffStackDatasetNormalized(Dataset):
    """
    Dataset with proper 16-bit normalization for microscopy images.
    Preserves the full dynamic range of 16-bit data.
    Supports structure-aware cropping if structure_map_path is provided.
    """
    
    def __init__(self, tiff_path, transform=None, max_frames=None, 
                 normalize_per_image=True, global_min=None, global_max=None,
                 structure_map_path=None, crop_size=None):
        self.tiff_path = tiff_path
        self.transform = transform
        self.normalize_per_image = normalize_per_image
        self.global_min = global_min
        self.global_max = global_max
        self.crop_size = crop_size
        
        # Load structure map if provided
        self.structure_map = None
        if structure_map_path and os.path.exists(structure_map_path):
            print(f"Loading structure map from {structure_map_path}")
            self.structure_map = np.load(structure_map_path)
            # Ensure structure map matches frame count later
        
        # Load TIFF and get number of frames
        with Image.open(tiff_path) as img:
            try:
                self.n_frames = img.n_frames
            except AttributeError:
                self.n_frames = 1
            
            self.img_size = img.size
            self.img_mode = img.mode
            
            print(f"Loaded TIFF: {tiff_path}")
            print(f"  Size: {self.img_size}")
            print(f"  Mode: {self.img_mode}")
            print(f"  Frames: {self.n_frames}")
            print(f"  Normalization: {'per-image' if normalize_per_image else 'global'}")
            
            # Compute global min/max if needed
            if not normalize_per_image and (global_min is None or global_max is None):
                print("  Computing global min/max...")
                all_mins = []
                all_maxs = []
                for i in range(min(self.n_frames, 100)):  # Sample first 100 frames
                    img.seek(i)
                    frame = np.array(img)
                    all_mins.append(frame.min())
                    all_maxs.append(frame.max())
                self.global_min = np.min(all_mins)
                self.global_max = np.max(all_maxs)
                print(f"  Global range: [{self.global_min}, {self.global_max}]")
        
        if max_frames is not None:
            self.n_frames = min(self.n_frames, max_frames)
        
        self.num_classes = 1
    
    def __len__(self):
        return self.n_frames
    
    def __getitem__(self, idx):
        if idx >= self.n_frames:
            raise IndexError(f"Frame index {idx} out of range [0, {self.n_frames})")
        
        # Open TIFF and seek to the requested frame
        with Image.open(self.tiff_path) as img:
            img.seek(idx)
            frame_array = np.array(img)
        
        # Normalize to 0-255 range
        if self.normalize_per_image:
            vmin, vmax = frame_array.min(), frame_array.max()
        else:
            vmin, vmax = self.global_min, self.global_max
        
        frame_array = ((frame_array - vmin) / (vmax - vmin + 1e-8) * 255).astype(np.uint8)
        
        # Convert to RGB (replicate channel)
        frame = Image.fromarray(frame_array, mode='L').convert('RGB')
        
        # Perform structure-aware cropping if enabled
        if self.structure_map is not None and self.crop_size is not None:
            # Get structure map for this frame
            # Handle case where structure map might have fewer frames or mismatch
            s_idx = idx % len(self.structure_map)
            skeleton = self.structure_map[s_idx]
            
            W, H = frame.size
            crop_h, crop_w = self.crop_size, self.crop_size
            
            if W >= crop_w and H >= crop_h:
                best_crop = None
                max_structure = -1
                
                # Try N times to find a good crop
                for _ in range(10):
                    x = np.random.randint(0, W - crop_w + 1)
                    y = np.random.randint(0, H - crop_h + 1)
                    
                    # Check structure content
                    skel_crop = skeleton[y:y+crop_h, x:x+crop_w]
                    structure_count = np.sum(skel_crop > 0)
                    
                    if structure_count > max_structure:
                        max_structure = structure_count
                        best_crop = (x, y, crop_w, crop_h)
                        
                    # If we found something decent (e.g. > 1% pixels are structure), stop
                    if structure_count > (crop_w * crop_h * 0.01):
                        break
                
                # Apply the best crop found
                if best_crop:
                    x, y, w, h = best_crop
                    frame = frame.crop((x, y, x+w, y+h))
        
        # Apply transforms if provided
        if self.transform is not None:
            frame = self.transform(frame)
        
        return frame, 0
