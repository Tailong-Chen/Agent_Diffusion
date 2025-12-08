#!/bin/bash

# Training script for JiT on 1024x1024 TIFF stack data
# Recommended configuration for single GPU or multi-GPU training

# Model: JiT-B/32 (recommended for 1024x1024 images)
# Patch size 32 creates 32x32 = 1024 patches (manageable sequence length)
# Patch size 16 would create 64x64 = 4096 patches (very long sequence)

# Adjust these parameters based on your GPU memory:
# - For 24GB GPU: batch_size=8-16
# - For 40GB GPU: batch_size=16-32
# - For 80GB GPU: batch_size=32-64

# Single GPU training
python main_jit.py \
--model JiT-B/32 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 1024 --noise_scale 2.0 \
--batch_size 16 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 16 --num_images 1000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ./output_tiff_1024 --resume ./output_tiff_1024 \
--data_path ./Data --tiff_file mt.tif --use_tiff \
--eval_freq 50

# Note: Disabled online_eval by default since we don't have reference statistics
# To enable evaluation, add: --online_eval
# But you'll need to create reference statistics first or modify evaluation code

# Multi-GPU training (8 GPUs example)
# Uncomment and adjust for your setup:
# torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
# main_jit.py \
# --model JiT-B/32 \
# --proj_dropout 0.0 \
# --P_mean -0.8 --P_std 0.8 \
# --img_size 1024 --noise_scale 2.0 \
# --batch_size 16 --blr 5e-5 \
# --epochs 600 --warmup_epochs 5 \
# --gen_bsz 16 --num_images 1000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
# --output_dir ./output_tiff_1024 --resume ./output_tiff_1024 \
# --data_path ./Data --tiff_file mt.tif --use_tiff \
# --eval_freq 50
