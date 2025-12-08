#!/bin/bash

# 指定使用的 GPU ID (例如: 0,1)
export CUDA_VISIBLE_DEVICES=0,1

# 设置使用的GPU数量
NUM_GPUS=2

# 设置数据路径和参数
# 假设数据在 ./Data/mt.tif
TIFF_FILE="mt.tif"
DATA_PATH="./Data"
OUTPUT_DIR="./output_jit_unet_ddp_mt"

# 训练参数
BATCH_SIZE=1      # 单卡 Batch Size (总 Batch Size = NUM_GPUS * BATCH_SIZE)
EPOCHS=5000000
LR=1e-4

# 打印信息
echo "🚀 开始多卡训练 (JiT-UNet)..."
echo "使用的 GPU 数量: $NUM_GPUS"
echo "数据文件: $DATA_PATH/$TIFF_FILE"
echo "输出目录: $OUTPUT_DIR"

# 自动检测断点续训
RESUME_ARGS=""
if [ -f "$OUTPUT_DIR/checkpoint-last.pth" ]; then
    echo "🔄 检测到上次的检查点，将恢复训练..."
    RESUME_ARGS="--resume $OUTPUT_DIR"
else
    echo "🆕 未检测到检查点，将开始新训练..."
fi

# 使用 torchrun 启动分布式训练
# --nproc_per_node: 使用的 GPU 数量
# --master_port: 防止端口冲突，随机指定一个
torchrun --nproc_per_node=$NUM_GPUS --master_port=29505 main_jit_unet.py \
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
    --num_workers 8 \
    --save_last_freq 200 \
    --eval_freq 200 \
    --online_eval \
    --gen_bsz 1 \
    --img_size 1024 \
    --accum_iter 4 \
    $RESUME_ARGS

# 训练完成后提示
echo "✅ 训练完成！结果保存在 $OUTPUT_DIR"
