#!/bin/bash

# 指定使用的 GPU ID (例如: 0,1)
export CUDA_VISIBLE_DEVICES=4,5

# 设置使用的GPU数量
NUM_GPUS=2

# 设置数据路径和参数
# 假设数据在 ./Data/mt.tif
TIFF_FILE="Dark-mitochondrion_1024.tif"
DATA_PATH="./Data"
OUTPUT_DIR="./output_jit_unet_ddp_patch256_mitochondrion"

# 训练参数
# 256x256 图像较小，可以增大 Batch Size
BATCH_SIZE=8      # 单卡 Batch Size
EPOCHS=5000000
LR=1e-4
IMG_SIZE=256       # Patch Size

# 打印信息
echo "🚀 开始 JiT-UNet 分布式训练 (Patch Training)"
echo "   GPUs: $NUM_GPUS (IDs: $CUDA_VISIBLE_DEVICES)"
echo "   Data: $DATA_PATH/$TIFF_FILE"
echo "   Output: $OUTPUT_DIR"
echo "   Batch Size: $BATCH_SIZE (Total: $((BATCH_SIZE * NUM_GPUS)))"
echo "   Image Size: $IMG_SIZE (Patch)"

# 预处理结构信息
echo "🔍 正在预处理结构信息..."
python preprocess_structure.py --tiff_path "$DATA_PATH/$TIFF_FILE" --output_path "$DATA_PATH/mt_skeletons.npy"

# 检查是否存在 checkpoint-last.pth，如果存在则自动恢复训练
if [ -f "$OUTPUT_DIR/checkpoint-last.pth" ]; then
    echo "🔄 检测到上次训练的 Checkpoint，将恢复训练..."
    RESUME_ARGS="--resume $OUTPUT_DIR"
else
    echo "🆕 未检测到 Checkpoint，将开始新训练..."
    RESUME_ARGS=""
fi

# 运行 torchrun
# 注意：--nproc_per_node 必须等于使用的 GPU 数量
# 设置 num_workers=0 以避免多进程导致的显存问题
torchrun --nproc_per_node=$NUM_GPUS --master_port=29504 main_jit_unet.py \
    --model UNet \
    --tiff_file "$TIFF_FILE" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --epochs $EPOCHS \
    --use_tiff \
    --use_normalized_tiff \
    --normalize_per_image \
    --num_workers 16 \
    --save_last_freq 20 \
    --eval_freq 200 \
    --online_eval \
    --gen_bsz 8 \
    --img_size $IMG_SIZE \
    --accum_iter 1 \
    $RESUME_ARGS

# 训练完成后提示
echo "✅ 训练完成！结果保存在 $OUTPUT_DIR"
