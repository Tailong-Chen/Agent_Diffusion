import os
import numpy as np
import tifffile

def process_tiff():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, 'Data/Dark-mitochondrion.tif')
    output_path = os.path.join(base_dir, 'Data/Dark-mitochondrion_1024.tif')

    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    print(f"Reading {input_path}...")
    img_stack = tifffile.imread(input_path)
    print(f"Original shape: {img_stack.shape}")

    # Ensure it's (Frames, H, W)
    if len(img_stack.shape) == 2:
        img_stack = img_stack[np.newaxis, ...]
    
    frames, h, w = img_stack.shape
    
    if h != 2048 or w != 2048:
        print(f"Warning: Expected 2048x2048, got {h}x{w}. Proceeding with 1024x1024 crops anyway.")

    new_patches = []
    
    # Crop logic
    # 0,0  | 0,1
    # -----+-----
    # 1,0  | 1,1
    
    for i in range(frames):
        frame = img_stack[i]
        
        # Top-Left
        p1 = frame[0:1024, 0:1024]
        # Top-Right
        p2 = frame[0:1024, 1024:2048]
        # Bottom-Left
        p3 = frame[1024:2048, 0:1024]
        # Bottom-Right
        p4 = frame[1024:2048, 1024:2048]
        
        new_patches.extend([p1, p2, p3, p4])

    new_stack = np.array(new_patches)
    print(f"New shape: {new_stack.shape}")
    
    print(f"Saving to {output_path}...")
    tifffile.imwrite(output_path, new_stack)
    print("Done.")

if __name__ == "__main__":
    process_tiff()
