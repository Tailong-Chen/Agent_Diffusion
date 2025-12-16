#!/bin/bash

# Define the base directory
BASE_DIR="/home/ctl/code/Agent_Diffusion/JiT"
NUM_IMAGES=10
BATCH_SIZE=4 # 显存允许的情况下可以调大，生成大图建议调小
GPUS=1

# 使用 ER 结构作为示例
DIR_NAME="output_jit_unet_ddp_patch256_ER"
CKPT_PATH="$BASE_DIR/$DIR_NAME/checkpoint-last.pth"
OUTPUT_DIR="$BASE_DIR/$DIR_NAME/generated_samples_1024"

echo "========================================================"
echo "Generating 1024x1024 images for ER using Sliding Window..."
echo "Checkpoint: $CKPT_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================================"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found at $CKPT_PATH"
    exit 1
fi

# 关键参数:
# --img_size 256: 模型训练时的分辨率 (Patch Size)
# --target_img_size 1024: 想要生成的目标分辨率
# 当 target_img_size > img_size 时，代码会自动启用 Overlap-Tile Sampling (滑动窗口)

torchrun --nproc_per_node=$GPUS --master_port=29506 "$BASE_DIR/sample_jit_unet.py" \
    --ckpt "$CKPT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_images $NUM_IMAGES \
    --batch_size $BATCH_SIZE \
    --img_size 256 \
    --target_img_size 1024 \
    --sampling_method "heun" \
    --num_sampling_steps 50

echo "Finished sampling."
