#!/bin/bash

# Define the base directory
BASE_DIR="/home/ctl/code/Agent_Diffusion/JiT"
# 和 run_sampling_all_structures.sh 保持一致，生成 2000 张
NUM_IMAGES=20 
# 生成 1024x1024 大图时显存占用较大，建议 Batch Size 设小一点 (例如 1-4)
BATCH_SIZE=4 
GPUS=4

# List of structures and their corresponding output directories
declare -A STRUCTURES
STRUCTURES["ER"]="output_jit_unet_ddp_patch256_ER"
STRUCTURES["F_actin"]="output_jit_unet_ddp_patch256_F_actin"
STRUCTURES["MT"]="output_jit_unet_ddp_patch256_MT"
STRUCTURES["Thy"]="output_jit_unet_ddp_patch256_Thy"

# Loop through each structure
for KEY in "${!STRUCTURES[@]}"; do
    DIR_NAME="${STRUCTURES[$KEY]}"
    CKPT_PATH="$BASE_DIR/$DIR_NAME/checkpoint-last.pth"
    # 输出目录区分 1024
    OUTPUT_DIR="$BASE_DIR/$DIR_NAME/generated_samples_1024"

    echo "========================================================"
    echo "Processing $KEY (1024x1024)..."
    echo "Checkpoint: $CKPT_PATH"
    echo "Output Directory: $OUTPUT_DIR"
    echo "========================================================"

    if [ ! -f "$CKPT_PATH" ]; then
        echo "Error: Checkpoint not found at $CKPT_PATH"
        continue
    fi

    # Run sampling
    # 使用 target_img_size 1024 触发滑动窗口生成
    torchrun --nproc_per_node=$GPUS --master_port=29507 "$BASE_DIR/sample_jit_unet.py" \
        --ckpt "$CKPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_images $NUM_IMAGES \
        --batch_size $BATCH_SIZE \
        --img_size 256 \
        --target_img_size 1024 \
        --sampling_method "heun" \
        --num_sampling_steps 50

    echo "Finished sampling for $KEY"
    echo ""
done

echo "All tasks completed."
