#!/bin/bash
# 快速启动脚本 - LDM训练
# 使用前请修改TIFF_PATH为您的数据路径

set -e  # 遇到错误立即退出

echo "=================================================="
echo "  TIFF堆栈LDM训练 - 快速启动"
echo "=================================================="

# ============================================
# 配置区域 - 请修改这里
# ============================================

# 您的TIFF文件路径（必须修改）
TIFF_PATH="./data/mt.tif"

# 输出目录
OUTPUT_DIR="./output_ldm_mt"

# 图像尺寸 (512 或 1024)
IMAGE_SIZE=1024

# 批次大小 (1024用2, 512用6)
BATCH_SIZE=2

# 是否只训练AutoEncoder (true/false)
ONLY_AE=false

# 是否只训练Diffusion (true/false)
ONLY_DIFF=false

# AutoEncoder checkpoint路径 (如果ONLY_DIFF=true)
AE_CHECKPOINT=""

# ============================================
# 检查依赖
# ============================================

echo ""
echo "1️⃣  检查Python依赖..."

if ! python -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch未安装"
    echo "请运行: pip install -r requirements_tiff_ldm.txt"
    exit 1
fi

if ! python -c "import monai" 2>/dev/null; then
    echo "❌ MONAI未安装"
    echo "请运行: pip install -r requirements_tiff_ldm.txt"
    exit 1
fi

if ! python -c "from generative.networks.nets import AutoencoderKL" 2>/dev/null; then
    echo "❌ MONAI GenerativeModels未安装"
    echo "请运行: pip install -r requirements_tiff_ldm.txt"
    exit 1
fi

echo "✅ 依赖检查通过"

# ============================================
# 检查TIFF文件
# ============================================

echo ""
echo "2️⃣  检查TIFF数据..."

if [ ! -f "$TIFF_PATH" ]; then
    echo "❌ TIFF文件不存在: $TIFF_PATH"
    echo "请修改脚本中的TIFF_PATH变量"
    exit 1
fi

echo "✅ TIFF文件存在: $TIFF_PATH"

# 运行数据检查
python check_tiff_data.py --tiff_path "$TIFF_PATH"

read -p "是否继续训练? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "训练已取消"
    exit 0
fi

# ============================================
# 开始训练
# ============================================

echo ""
echo "3️⃣  开始训练..."
echo ""

# 构建训练命令
TRAIN_CMD="python train_tiff_ldm.py \
    --tiff_path $TIFF_PATH \
    --output_dir $OUTPUT_DIR \
    --image_size $IMAGE_SIZE \
    --batch_size $BATCH_SIZE"

# 添加可选参数
if [ "$ONLY_AE" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --skip_diffusion"
    echo "📝 模式: 仅训练AutoEncoder"
fi

if [ "$ONLY_DIFF" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --skip_autoencoder"
    if [ -n "$AE_CHECKPOINT" ]; then
        TRAIN_CMD="$TRAIN_CMD --autoencoder_checkpoint $AE_CHECKPOINT"
    fi
    echo "📝 模式: 仅训练Diffusion"
fi

echo "📝 训练配置:"
echo "   - TIFF文件: $TIFF_PATH"
echo "   - 输出目录: $OUTPUT_DIR"
echo "   - 图像尺寸: ${IMAGE_SIZE}×${IMAGE_SIZE}"
echo "   - 批次大小: $BATCH_SIZE"
echo ""
echo "🚀 执行命令:"
echo "$TRAIN_CMD"
echo ""

# 执行训练
eval $TRAIN_CMD

# ============================================
# 训练完成
# ============================================

echo ""
echo "=================================================="
echo "  🎉 训练完成！"
echo "=================================================="
echo ""
echo "📁 输出文件:"
echo "   - Checkpoints: $OUTPUT_DIR/checkpoints/"
echo "   - 样本图像: $OUTPUT_DIR/samples/"
echo "   - 训练曲线: $OUTPUT_DIR/training_history.png"
echo ""
echo "🎨 生成新样本:"
echo "python generate_samples.py \\"
echo "    --checkpoint $OUTPUT_DIR/checkpoints/diffusion_epoch_250.pth \\"
echo "    --num_samples 20 \\"
echo "    --output_dir ./generated"
echo ""
echo "=================================================="

