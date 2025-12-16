#!/bin/bash

# Define the base directory
BASE_DIR="/home/ctl/code/Agent_Diffusion/JiT"
GPUS=6 # Adjust based on available GPUs

# Structure specific settings for Mitochondria (MT)
STRUCTURE="MT"
CKPT_DIR="output_jit_unet_ddp_patch256_mitochondrion"
CKPT_PATH="$BASE_DIR/$CKPT_DIR/checkpoint-last.pth"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found at $CKPT_PATH"
    exit 1
fi

# Function to run sampling
run_sampling() {
    local NUM_IMAGES=$1
    local SIZE=$2
    local OUTPUT_SUBDIR="generated_samples_${SIZE}x${SIZE}_${NUM_IMAGES}"
    local OUTPUT_DIR="$BASE_DIR/$CKPT_DIR/$OUTPUT_SUBDIR"

    echo "========================================================"
    echo "Sampling $NUM_IMAGES images of size $SIZE x $SIZE for $STRUCTURE..."
    echo "Checkpoint: $CKPT_PATH"
    echo "Output Directory: $OUTPUT_DIR"
    echo "========================================================"

    # Adjust batch size based on resolution to avoid OOM
    # These values are estimates, adjust if OOM occurs
    if [ "$SIZE" -eq 256 ]; then
        BATCH_SIZE=64
    elif [ "$SIZE" -eq 512 ]; then
        BATCH_SIZE=16
    elif [ "$SIZE" -eq 1024 ]; then
        BATCH_SIZE=4
    elif [ "$SIZE" -eq 2048 ]; then
        BATCH_SIZE=1
    else
        BATCH_SIZE=1
    fi

    # Run sampling
    # Using a different master_port to avoid conflicts
    torchrun --nproc_per_node=$GPUS --master_port=29506 "$BASE_DIR/sample_jit_unet.py" \
        --ckpt "$CKPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_images $NUM_IMAGES \
        --batch_size $BATCH_SIZE \
        --img_size 256 \
        --target_img_size $SIZE \
        --sampling_method "heun" \
        --num_sampling_steps 50

    echo "Finished sampling $SIZE x $SIZE"
    echo ""
}

# 1. 2000 images 256x256
run_sampling 2000 256

# 2. 200 images 512x512
run_sampling 200 512

# 3. 200 images 1024x1024
run_sampling 200 1024

# 4. 200 images 2048x2048
run_sampling 200 2048

echo "All sampling tasks completed for Mitochondria (MT)."
