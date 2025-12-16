#!/bin/bash

# Define the base directory
BASE_DIR="/home/ctl/code/Agent_Diffusion/JiT"
NUM_IMAGES=2000
BATCH_SIZE=64 # Adjust based on GPU memory
GPUS=4

# List of structures and their corresponding output directories
declare -A STRUCTURES
STRUCTURES["CCP"]="output_jit_unet_ddp_patch256_CCP"
#STRUCTURES["F_actin"]="output_jit_unet_ddp_patch256_F_actin"
#STRUCTURES["MT"]="output_jit_unet_ddp_patch256_MT"
#STRUCTURES["Thy"]="output_jit_unet_ddp_patch256_Thy"

# Loop through each structure
for KEY in "${!STRUCTURES[@]}"; do
    DIR_NAME="${STRUCTURES[$KEY]}"
    CKPT_PATH="$BASE_DIR/$DIR_NAME/checkpoint-last.pth"
    OUTPUT_DIR="$BASE_DIR/$DIR_NAME/generated_samples_2000"

    echo "========================================================"
    echo "Processing $KEY..."
    echo "Checkpoint: $CKPT_PATH"
    echo "Output Directory: $OUTPUT_DIR"
    echo "========================================================"

    if [ ! -f "$CKPT_PATH" ]; then
        echo "Error: Checkpoint not found at $CKPT_PATH"
        continue
    fi

    # Run sampling
    torchrun --nproc_per_node=$GPUS --master_port=29505 "$BASE_DIR/sample_jit_unet.py" \
        --ckpt "$CKPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --num_images $NUM_IMAGES \
        --batch_size $BATCH_SIZE \
        --img_size 256 \
        --target_img_size 256 \
        --sampling_method "heun" \
        --num_sampling_steps 50

    echo "Finished sampling for $KEY"
    echo ""
done

echo "All tasks completed."
