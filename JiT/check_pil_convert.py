import numpy as np
from PIL import Image
import os

# Create a dummy 16-bit TIFF
data = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
img = Image.fromarray(data)
img.save("test_16bit.tif")

# Method 1: TiffStackDataset approach
with Image.open("test_16bit.tif") as img:
    img_rgb = img.convert("RGB")
    arr_rgb = np.array(img_rgb)
    print(f"Method 1 (convert RGB) mean: {arr_rgb.mean()}")
    print(f"Method 1 min/max: {arr_rgb.min()}/{arr_rgb.max()}")

# Method 2: calc_fid_tiff.py approach
with Image.open("test_16bit.tif") as img:
    arr = np.array(img)
    arr = arr.astype(float)
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
    img_norm = Image.fromarray(arr.astype(np.uint8))
    img_norm_rgb = img_norm.convert("RGB")
    arr_norm = np.array(img_norm_rgb)
    print(f"Method 2 (min-max) mean: {arr_norm.mean()}")
    print(f"Method 2 min/max: {arr_norm.min()}/{arr_norm.max()}")

os.remove("test_16bit.tif")
