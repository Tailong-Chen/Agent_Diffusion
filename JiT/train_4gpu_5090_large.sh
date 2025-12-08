#!/bin/bash

# Training script for JiT-L/32 on 1024x1024 TIFF with 4x RTX 5090 GPUs
# Larger model with higher capacity - uses more VRAM

echo "============================================================"
echo "JiT-L Training on 4x RTX 5090 GPUs (Large Model)"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  - Model: JiT-L/32 (304M parameters)"
echo "  - Image Size: 1024x1024"
echo "  - Batch Size: 48 per GPU (Total: 192)"
echo "  - GPUs: 4"
echo "  - Expected VRAM: ~28GB per GPU"
echo ""
echo "Starting training..."
echo ""

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-L/32 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 1024 --noise_scale 2.0 \
--batch_size 48 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 48 --num_images 1000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ./output_tiff_1024_large --resume ./output_tiff_1024_large \
--data_path ./Data --tiff_file mt.tif --use_tiff \
--eval_freq 50 \
--num_workers 8

echo ""
echo "============================================================"
echo "Training completed!"
echo "============================================================"
