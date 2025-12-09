from PIL import Image
import numpy as np

try:
    with Image.open("Data/mt.tif") as img:
        print(f"Mode: {img.mode}")
        img.seek(0)
        arr = np.array(img)
        print(f"Min: {arr.min()}, Max: {arr.max()}, Mean: {arr.mean()}")
        
        # Check what convert('RGB') does
        img_rgb = img.convert("RGB")
        arr_rgb = np.array(img_rgb)
        print(f"Convert RGB Min: {arr_rgb.min()}, Max: {arr_rgb.max()}, Mean: {arr_rgb.mean()}")
except Exception as e:
    print(e)
