import os
import numpy as np
import tifffile
import cv2
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def preprocess_structure(tiff_path, output_path, vis_path):
    print(f"Processing {tiff_path}...")
    
    # Read TIFF
    # Use tifffile for reliable reading of 16-bit stacks
    img_stack = tifffile.imread(tiff_path)
    
    print(f"Data shape: {img_stack.shape}, dtype: {img_stack.dtype}")
    
    skeletons = []
    
    # Create visualization grid
    n_vis = min(5, len(img_stack))
    fig, axes = plt.subplots(n_vis, 3, figsize=(15, 5*n_vis))
    if n_vis == 1: axes = [axes]
    
    for i in tqdm(range(len(img_stack))):
        img = img_stack[i]
        
        # Normalize to 0-255 for thresholding
        # Robust normalization using percentiles to ignore outliers
        p1, p99 = np.percentile(img, (1, 99))
        img_norm = np.clip((img - p1) / (p99 - p1 + 1e-8), 0, 1)
        
        # Thresholding
        try:
            thresh = threshold_otsu(img_norm)
            binary = img_norm > thresh
        except Exception:
            # Fallback if image is uniform
            binary = img_norm > 0.5
            
        # Skeletonize
        # skeletonize expects boolean
        skeleton = skeletonize(binary)
        
        skeletons.append(skeleton)
        
        # Visualization for first few frames
        if i < n_vis:
            ax = axes[i] if n_vis > 1 else axes
            ax[0].imshow(img, cmap='gray')
            ax[0].set_title(f"Original Frame {i}")
            ax[1].imshow(binary, cmap='gray')
            ax[1].set_title("Binary Threshold")
            ax[2].imshow(skeleton, cmap='gray')
            ax[2].set_title("Skeleton")
            
    # Stack and save
    skeletons = np.stack(skeletons).astype(np.uint8) * 255 # Save as 0/255 uint8
    
    print(f"Saving skeletons to {output_path}...")
    np.save(output_path, skeletons)
    
    print(f"Saving visualization to {vis_path}...")
    plt.tight_layout()
    plt.savefig(vis_path)
    plt.close()
    
    # Calculate statistics
    pixel_counts = np.sum(skeletons > 0, axis=(1, 2))
    print(f"Skeleton pixel stats - Mean: {pixel_counts.mean():.2f}, Min: {pixel_counts.min()}, Max: {pixel_counts.max()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiff_path', type=str, default='./Data/mt.tif')
    parser.add_argument('--output_path', type=str, default='./Data/mt_skeletons.npy')
    parser.add_argument('--vis_path', type=str, default='./Data/structure_vis.png')
    args = parser.parse_args()
    
    preprocess_structure(args.tiff_path, args.output_path, args.vis_path)
