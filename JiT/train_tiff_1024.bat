@echo off
REM Training script for JiT on 1024x1024 TIFF stack data (Windows)
REM Recommended configuration for single GPU or multi-GPU training

REM Model: JiT-B/32 (recommended for 1024x1024 images)
REM Patch size 32 creates 32x32 = 1024 patches (manageable sequence length)

REM Adjust batch_size based on your GPU memory:
REM - For 24GB GPU: batch_size=8-16
REM - For 40GB GPU: batch_size=16-32
REM - For 80GB GPU: batch_size=32-64

echo Starting JiT training on TIFF stack data...
echo.

REM Single GPU training
python main_jit.py ^
--model JiT-B/32 ^
--proj_dropout 0.0 ^
--P_mean -0.8 --P_std 0.8 ^
--img_size 1024 --noise_scale 2.0 ^
--batch_size 16 --blr 5e-5 ^
--epochs 600 --warmup_epochs 5 ^
--gen_bsz 16 --num_images 1000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 ^
--output_dir ./output_tiff_1024 --resume ./output_tiff_1024 ^
--data_path ./Data --tiff_file mt.tif --use_tiff ^
--eval_freq 50

echo.
echo Training completed!
pause

REM Multi-GPU training (8 GPUs example)
REM Uncomment and adjust for your setup:
REM torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 ^
REM main_jit.py ^
REM --model JiT-B/32 ^
REM --proj_dropout 0.0 ^
REM --P_mean -0.8 --P_std 0.8 ^
REM --img_size 1024 --noise_scale 2.0 ^
REM --batch_size 16 --blr 5e-5 ^
REM --epochs 600 --warmup_epochs 5 ^
REM --gen_bsz 16 --num_images 1000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 ^
REM --output_dir ./output_tiff_1024 --resume ./output_tiff_1024 ^
REM --data_path ./Data --tiff_file mt.tif --use_tiff ^
REM --eval_freq 50
