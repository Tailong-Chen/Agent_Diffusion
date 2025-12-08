from PIL import Image
import numpy as np

# Load and inspect TIFF file
img = Image.open('Data/mt.tif')
print(f'Format: {img.format}')
print(f'Mode: {img.mode}')
print(f'Size: {img.size}')

# Check if it's a multi-frame TIFF
try:
    n_frames = img.n_frames
    print(f'Number of frames: {n_frames}')
    
    # Check first few frames
    for i in range(min(3, n_frames)):
        img.seek(i)
        frame = np.array(img)
        print(f'Frame {i}: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}')
except Exception as e:
    print(f'Single frame TIFF or error: {e}')
    frame = np.array(img)
    print(f'Image shape: {frame.shape}, dtype: {frame.dtype}')
