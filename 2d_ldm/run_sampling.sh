#!/bin/bash

# 设置通用参数
IMAGE_SIZE=512
TOTAL_SAMPLES=1000
BATCH_SIZE=5 # 每张卡的batch size
STEPS=1000

# 创建日志目录
mkdir -p ./generated_samples

echo "🚀 Starting DDP parallel sampling..."

# 1. ER (使用 GPU 0,1)
echo "   [Task 1] ER -> GPUs 0,1"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29500 sample_ddpm_ddp.py \
    --model_dir ./output_ddpm_512_v2_ER \
    --output_dir ./generated_samples/ER \
    --image_size $IMAGE_SIZE \
    --total_samples $TOTAL_SAMPLES \
    --batch_size $BATCH_SIZE \
    --steps $STEPS > ./generated_samples/log_ER.txt 2>&1 &
PID_ER=$!

# 2. F-actin (使用 GPU 2,3)
echo "   [Task 2] F-actin -> GPUs 2,3"
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29501 sample_ddpm_ddp.py \
    --model_dir ./output_ddpm_512_v2_F-actin \
    --output_dir ./generated_samples/F-actin \
    --image_size $IMAGE_SIZE \
    --total_samples $TOTAL_SAMPLES \
    --batch_size $BATCH_SIZE \
    --steps $STEPS > ./generated_samples/log_F-actin.txt 2>&1 &
PID_FACTIN=$!

# 3. MT (使用 GPU 4,5)
echo "   [Task 3] MT -> GPUs 4,5,"
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29502 sample_ddpm_ddp.py \
    --model_dir ./output_ddpm_512_v2_mt \
    --output_dir ./generated_samples/mt \
    --image_size $IMAGE_SIZE \
    --total_samples $TOTAL_SAMPLES \
    --batch_size $BATCH_SIZE \
    --steps $STEPS > ./generated_samples/log_mt.txt 2>&1 &
PID_MT=$!

echo "✅ All DDP tasks submitted."
echo "   ER PID: $PID_ER (Port 29500)"
echo "   F-actin PID: $PID_FACTIN (Port 29501)"
echo "   MT PID: $PID_MT (Port 29502)"
echo "   Logs: ./generated_samples/log_*.txt"
echo "⏳ Waiting for completion..."

wait

echo "🎉 All DDP sampling tasks completed!"
