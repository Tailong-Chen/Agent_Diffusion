import argparse
import os
import shutil
import tempfile
import numpy as np
from PIL import Image
import torch_fidelity
from tqdm import tqdm

def extract_tiff_frames(tiff_path, output_dir):
    """Extracts frames from a TIFF stack and saves them as PNGs."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Extracting frames from {tiff_path} to {output_dir}...")
    
    try:
        with Image.open(tiff_path) as img:
            n_frames = getattr(img, 'n_frames', 1)
            
            for i in tqdm(range(n_frames)):
                img.seek(i)
                # Convert to 8-bit if necessary
                frame = img.copy()
                
                # Normalize 16-bit to 8-bit if needed
                if frame.mode == 'I;16' or frame.mode == 'I;16B':
                    arr = np.array(frame)
                    # Normalize to 0-255
                    arr = arr.astype(float)
                    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
                    frame = Image.fromarray(arr.astype(np.uint8))
                elif frame.mode == 'I':
                     arr = np.array(frame)
                     arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
                     frame = Image.fromarray(arr.astype(np.uint8))

                # Convert to RGB for Inception (FID)
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                
                frame.save(os.path.join(output_dir, f"frame_{i:05d}.png"))
                
    except Exception as e:
        print(f"Error extracting TIFF frames: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Calculate FID between generated images and TIFF training data')
    parser.add_argument('--ref_tiff', type=str, required=True, help='Path to reference TIFF stack (training data)')
    parser.add_argument('--gen_dir', type=str, required=True, help='Path to directory containing generated images')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for FID calculation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create temporary directory for reference images
    with tempfile.TemporaryDirectory() as temp_ref_dir:
        # Extract TIFF frames
        extract_tiff_frames(args.ref_tiff, temp_ref_dir)
        
        print(f"Calculating FID between {temp_ref_dir} and {args.gen_dir}...")
        
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=args.gen_dir,
            input2=temp_ref_dir,
            cuda=args.device == 'cuda',
            isc=False,
            fid=True,
            kid=False,
            verbose=True,
            batch_size=args.batch_size
        )
        
        print(f"FID: {metrics_dict['frechet_inception_distance']}")

if __name__ == '__main__':
    main()
