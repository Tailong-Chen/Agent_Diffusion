"""
DDPM Sampling Script (FP32 Companion)
====================================

本腳本與 train_ddpm_simple.py 的 FP32 訓練流程配套，用於：
1. 從已訓練的 DDPM checkpoint 生成樣本，保持純 FP32 數值路徑。
2. 每個樣本輸出最小值、最大值與平均值，協助監控是否出現全黑或 NaN。
3. 保存單張樣本與樣本网格，方便快速檢查訓練進度。

使用方式:
    python train_ddpm_simple_sampler.py \
        --checkpoint ./output_ddpm_256/checkpoints/best_model.pth \
        --output_dir ./output_ddpm_256/sampling_runs/run_001 \
        --image_size 256 --num_samples 8 --num_inference_steps 1000
"""

import argparse
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from monai.utils import set_determinism
from tqdm import tqdm

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDPM sampling script (FP32, no AMP)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained checkpoint")
    parser.add_argument("--output_dir", type=str, default="./ddpm_samples", help="Directory to store outputs")
    parser.add_argument("--image_size", type=int, default=256, help="Spatial size used during training")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=1000,
        help="Number of DDPM inference steps (matches scheduler timesteps)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device, e.g., 'cuda:0' or 'cpu'")
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Display tqdm progress bar for every reverse diffusion loop",
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=4,
        help="Number of columns in the summary grid (rows auto-adjust)",
    )
    parser.add_argument(
        "--save_numpy",
        action="store_true",
        help="Save raw numpy arrays (float32) alongside PNG figures",
    )
    return parser.parse_args()


def prepare_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("⚠️ CUDA 不可用，將退回 CPU。")
        return torch.device("cpu")
    return torch.device(device_arg)


def build_unet(image_size: int, device: torch.device) -> torch.nn.Module:
    if image_size >= 512:
        num_channels = (128, 256, 512)
        attention_levels = (False, False, False)
        num_head_channels = 512
    else:
        num_channels = (128, 256, 256)
        attention_levels = (False, True, True)
        num_head_channels = 256

    unet = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_res_blocks=2,
        num_channels=num_channels,
        attention_levels=attention_levels,
        num_head_channels=num_head_channels,
    )
    return unet.to(device)


def load_checkpoint(unet: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> Tuple[int, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    unet.load_state_dict(state_dict)

    meta = {
        "epoch": ckpt.get("epoch"),
        "train_loss": ckpt.get("train_loss"),
        "val_loss": ckpt.get("val_loss"),
    }
    return unet, meta


def run_reverse_diffusion(
    unet: torch.nn.Module,
    scheduler: DDPMScheduler,
    image_size: int,
    device: torch.device,
    show_progress: bool,
) -> torch.Tensor:
    noise = torch.randn((1, 1, image_size, image_size), device=device).float()
    timesteps = scheduler.timesteps
    iterator = tqdm(timesteps, desc="Sampling", leave=False) if show_progress else timesteps

    sample = noise
    with torch.no_grad():
        for t in iterator:
            ts = torch.tensor((t,), device=device).long()
            model_out = unet(x=sample, timesteps=ts)
            step_result = scheduler.step(model_out, t, sample)
            sample = step_result[0] if isinstance(step_result, tuple) else step_result.prev_sample
    return sample


def tensor_to_numpy(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().cpu().numpy()
    return np.squeeze(array, axis=0).squeeze(axis=0)


def save_single_image(image_np: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(image_np, cmap="gray", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_tiff(image_np: np.ndarray, out_path: Path) -> None:
    """Save a single-channel image as a 16-bit TIFF.

    The script assumes `image_np` is in [0, 1]. We convert to uint16 for
    better dynamic range compatibility with common TIFF viewers.
    """
    arr = np.clip(image_np, 0.0, 1.0)
    arr16 = (arr * 65535.0).astype(np.uint16)
    img = Image.fromarray(arr16)
    img.save(str(out_path))


def save_grid(images: List[np.ndarray], cols: int, out_path: Path) -> None:
    if not images:
        return
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(rows, cols)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("off")
        if idx < len(images):
            ax.imshow(images[idx], cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_title(f"Sample {idx + 1}")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_stats(image_np: np.ndarray, prefix: str = "") -> Tuple[float, float, float]:
    d_min = float(image_np.min())
    d_max = float(image_np.max())
    d_mean = float(image_np.mean())
    tag = f"{prefix} " if prefix else ""
    print(f"{tag}Min: {d_min:.4f} | Max: {d_max:.4f} | Mean: {d_mean:.4f}")

    if d_max == 0.0 and d_min == 0.0:
        print(f"{tag}⚠️ 警告：生成結果為全0，請檢查模型或scheduler設定")
    if math.isnan(d_mean):
        print(f"{tag}⚠️ 警告：生成結果包含 NaN")
    return d_min, d_max, d_mean


def main():
    args = parse_args()
    set_determinism(args.seed)
    device = prepare_device(args.device)

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(exist_ok=True)

    print("=" * 70)
    print("🧪 DDPM 采樣（FP32 無 AMP）")
    print("=" * 70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Image size: {args.image_size}")
    print(f"Samples: {args.num_samples}")
    print(f"Inference steps: {args.num_inference_steps}")

    unet = build_unet(args.image_size, device)
    unet, meta = load_checkpoint(unet, checkpoint_path, device)
    unet.eval()

    if meta["epoch"] is not None:
        print(f"🔁 來源訓練 Epoch: {meta['epoch']}")
    if meta["val_loss"] is not None:
        print(f"📉 最佳 Val Loss: {meta['val_loss']:.6f}")

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)

    all_images: List[np.ndarray] = []
    stats_log: List[Tuple[float, float, float]] = []

    for idx in range(args.num_samples):
        sample_tensor = run_reverse_diffusion(
            unet=unet,
            scheduler=scheduler,
            image_size=args.image_size,
            device=device,
            show_progress=args.show_progress,
        )
        sample_np = tensor_to_numpy(sample_tensor)
        all_images.append(sample_np)

        stats = print_stats(sample_np, prefix=f"Sample {idx + 1}")
        stats_log.append(stats)

        sample_tif = output_dir / "samples" / f"sample_{idx + 1:03d}.tif"
        save_tiff(sample_np, sample_tif)

        if args.save_numpy:
            np.save(output_dir / "samples" / f"sample_{idx + 1:03d}.npy", sample_np.astype(np.float32))

    # 不再自动生成样本网格。脚本仅保存每张采样图为 TIFF。

    stats_array = np.array(stats_log)
    global_min, global_max, global_mean = stats_array[:, 0].min(), stats_array[:, 1].max(), stats_array[:, 2].mean()
    print("=" * 70)
    print("📊 整體統計")
    print(f"Min (all samples): {global_min:.4f}")
    print(f"Max (all samples): {global_max:.4f}")
    print(f"Mean (average of means): {global_mean:.4f}")
    print("=" * 70)
    print(f"✅ 采樣完成，輸出位於: {output_dir}")


if __name__ == "__main__":
    main()
