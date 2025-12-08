"""
TIFF数据检查工具
检查TIFF堆栈是否符合训练要求

使用方法:
    python check_tiff_data.py --tiff_path your_data.tif
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False
    from PIL import Image


def load_tiff(tiff_path):
    """加载TIFF文件"""
    if HAS_TIFFFILE:
        return tifffile.imread(tiff_path)
    else:
        # 使用PIL加载
        images = []
        img = Image.open(tiff_path)
        try:
            for i in range(1000):
                img.seek(i)
                images.append(np.array(img))
        except EOFError:
            pass
        return np.stack(images)


def check_tiff_data(tiff_path, visualize=True, output_dir=None):
    """检查TIFF数据"""
    print("="*60)
    print("📊 TIFF数据检查")
    print("="*60)
    
    tiff_path = Path(tiff_path)
    
    # 1. 检查文件是否存在
    if not tiff_path.exists():
        print(f"❌ 文件不存在: {tiff_path}")
        return False
    
    print(f"✅ 文件存在: {tiff_path}")
    print(f"   文件大小: {tiff_path.stat().st_size / 1024**2:.2f} MB")
    
    # 2. 加载数据
    print("\n📂 加载数据...")
    try:
        if HAS_TIFFFILE:
            print("   使用: tifffile")
        else:
            print("   使用: PIL (建议安装tifffile以获得更好的性能)")
        
        images = load_tiff(tiff_path)
        print(f"✅ 数据加载成功")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False
    
    # 3. 检查数据形状
    print(f"\n📐 数据信息:")
    print(f"   原始形状: {images.shape}")
    print(f"   数据类型: {images.dtype}")
    
    # 处理不同的形状
    if images.ndim == 2:
        images = images[np.newaxis, ...]
        print(f"   ⚠️  检测到单张图像，已转换为堆栈")
    elif images.ndim == 4:
        if images.shape[-1] in [1, 3, 4]:
            images = images[..., 0]
            print(f"   ⚠️  检测到多通道图像，已提取第一个通道")
    
    num_images, height, width = images.shape[:3]
    
    print(f"\n✅ 处理后形状: {images.shape}")
    print(f"   图像数量: {num_images}")
    print(f"   图像尺寸: {height} × {width}")
    
    # 4. 检查图像数量
    print(f"\n📊 图像数量检查:")
    if num_images < 10:
        print(f"   ⚠️  图像数量较少 ({num_images}张)")
        print(f"      建议至少30-50张以获得良好效果")
    elif num_images < 30:
        print(f"   ⚠️  图像数量偏少 ({num_images}张)")
        print(f"      可以训练，但效果可能一般")
    else:
        print(f"   ✅ 图像数量充足 ({num_images}张)")
    
    # 5. 检查图像尺寸
    print(f"\n📏 图像尺寸检查:")
    if height != width:
        print(f"   ⚠️  图像不是正方形 ({height}×{width})")
        print(f"      训练时会自动调整为正方形")
    
    if height == 1024 and width == 1024:
        print(f"   ✅ 标准1024×1024尺寸")
        print(f"      需要32GB显存，batch_size=2")
    elif height == 512 and width == 512:
        print(f"   ✅ 512×512尺寸")
        print(f"      32GB显存下推荐，batch_size=6")
    elif height > 1024 or width > 1024:
        print(f"   ⚠️  图像尺寸过大 ({height}×{width})")
        print(f"      建议降采样到1024×1024或512×512")
    elif height < 256 or width < 256:
        print(f"   ⚠️  图像尺寸较小 ({height}×{width})")
        print(f"      可能无法获得理想效果")
    else:
        print(f"   ✅ 合适的图像尺寸 ({height}×{width})")
    
    # 6. 检查数值范围
    print(f"\n🔢 数值范围检查:")
    vmin, vmax = images.min(), images.max()
    print(f"   最小值: {vmin}")
    print(f"   最大值: {vmax}")
    print(f"   均值: {images.mean():.2f}")
    print(f"   标准差: {images.std():.2f}")
    
    if vmin < 0:
        print(f"   ⚠️  存在负值，训练时会自动归一化")
    
    if vmax <= 1.0:
        print(f"   ✅ 数值已归一化到[0, 1]")
    elif vmax <= 255:
        print(f"   ✅ 标准8-bit图像[0, 255]")
    elif vmax <= 65535:
        print(f"   ✅ 16-bit图像[0, 65535]")
    else:
        print(f"   ⚠️  数值范围异常，请检查数据")
    
    # 7. 检查是否有异常值
    print(f"\n🔍 异常值检查:")
    num_zeros = np.sum(images == 0)
    num_saturated = np.sum(images == vmax)
    total_pixels = images.size
    
    print(f"   全0像素: {num_zeros} ({num_zeros/total_pixels*100:.2f}%)")
    print(f"   饱和像素: {num_saturated} ({num_saturated/total_pixels*100:.2f}%)")
    
    if num_zeros / total_pixels > 0.5:
        print(f"   ⚠️  过多0值像素，请检查数据")
    if num_saturated / total_pixels > 0.1:
        print(f"   ⚠️  过多饱和像素，图像可能过曝")
    
    # 8. 计算图像统计信息
    print(f"\n📈 每张图像的统计信息:")
    means = [images[i].mean() for i in range(min(5, num_images))]
    stds = [images[i].std() for i in range(min(5, num_images))]
    
    for i in range(min(5, num_images)):
        print(f"   图像 {i+1}: 均值={means[i]:.2f}, 标准差={stds[i]:.2f}")
    
    # 检查图像间的差异
    mean_of_means = np.mean(means)
    std_of_means = np.std(means)
    
    if std_of_means / mean_of_means > 0.5:
        print(f"   ⚠️  图像间差异较大，可能影响训练")
    else:
        print(f"   ✅ 图像间差异适中")
    
    # 9. 可视化
    if visualize:
        print(f"\n🖼️  生成可视化...")
        
        # 创建输出目录
        if output_dir is None:
            output_dir = tiff_path.parent / "tiff_check"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 可视化样本
        num_show = min(9, num_images)
        ncols = 3
        nrows = (num_show + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows*4))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_show):
            row = i // ncols
            col = i % ncols
            
            ax = axes[row, col]
            img = images[i]
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f"Image {i+1}\nMean: {img.mean():.1f}, Std: {img.std():.1f}")
            ax.axis('off')
        
        # 隐藏多余的子图
        for i in range(num_show, nrows * ncols):
            row = i // ncols
            col = i % ncols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        sample_path = output_dir / "sample_images.png"
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 样本图像已保存: {sample_path}")
        
        # 绘制直方图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 整体直方图
        axes[0].hist(images.flatten(), bins=100, alpha=0.7, color='blue')
        axes[0].set_title("Overall Pixel Value Distribution")
        axes[0].set_xlabel("Pixel Value")
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True, alpha=0.3)
        
        # 每张图像的均值分布
        all_means = [images[i].mean() for i in range(num_images)]
        axes[1].bar(range(num_images), all_means, alpha=0.7, color='green')
        axes[1].set_title("Mean Value per Image")
        axes[1].set_xlabel("Image Index")
        axes[1].set_ylabel("Mean Pixel Value")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        hist_path = output_dir / "statistics.png"
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✅ 统计图表已保存: {hist_path}")
    
    # 10. 训练建议
    print(f"\n💡 训练建议:")
    print(f"="*60)
    
    if num_images >= 30 and height >= 512 and width >= 512:
        print("✅ 数据质量良好，可以开始训练！")
        
        if height == 1024 and width == 1024:
            print("\n推荐命令（1024×1024）:")
            print(f"python train_tiff_ldm.py \\")
            print(f"    --tiff_path {tiff_path} \\")
            print(f"    --output_dir ./output_ldm \\")
            print(f"    --image_size 1024 \\")
            print(f"    --batch_size 2")
        else:
            print("\n推荐命令（512×512）:")
            print(f"python train_tiff_ldm.py \\")
            print(f"    --tiff_path {tiff_path} \\")
            print(f"    --output_dir ./output_ldm \\")
            print(f"    --image_size 512 \\")
            print(f"    --batch_size 6")
    else:
        print("⚠️  数据存在一些问题，建议：")
        
        if num_images < 30:
            print(f"  - 增加图像数量（当前{num_images}张，建议至少30张）")
        
        if height < 512 or width < 512:
            print(f"  - 使用更高分辨率的图像")
        
        print("\n可以先进行快速测试:")
        print(f"python train_tiff_ldm.py \\")
        print(f"    --tiff_path {tiff_path} \\")
        print(f"    --output_dir ./test_run \\")
        print(f"    --image_size 256 \\")
        print(f"    --batch_size 8")
    
    print("="*60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="检查TIFF数据是否适合训练")
    parser.add_argument("--tiff_path", type=str, required=True, help="TIFF文件路径")
    parser.add_argument("--no_visualize", action="store_true", help="不生成可视化")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    
    args = parser.parse_args()
    
    success = check_tiff_data(
        args.tiff_path,
        visualize=not args.no_visualize,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n✅ 检查完成")
    else:
        print("\n❌ 检查失败")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

