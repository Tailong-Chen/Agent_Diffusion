@echo off
REM Training script for JiT on 1024x1024 TIFF with 4x RTX 5090 GPUs
REM Optimized for 32GB VRAM per GPU

echo ============================================================
echo JiT Training on 4x RTX 5090 GPUs
echo ============================================================
echo.
echo Configuration:
echo   - Model: JiT-B/32
echo   - Image Size: 1024x1024
echo   - Batch Size: 32 per GPU (Total: 128)
echo   - GPUs: 4
echo   - Expected VRAM: ~22GB per GPU
echo.
echo Starting training...
echo.

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 ^
main_jit.py ^
--model JiT-B/32 ^
--proj_dropout 0.0 ^
--P_mean -0.8 --P_std 0.8 ^
--img_size 1024 --noise_scale 2.0 ^
--batch_size 32 --blr 5e-5 ^
--epochs 600 --warmup_epochs 5 ^
--gen_bsz 32 --num_images 1000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 ^
--output_dir ./output_tiff_1024 --resume ./output_tiff_1024 ^
--data_path ./Data --tiff_file mt.tif --use_tiff ^
--eval_freq 50 ^
--num_workers 8

echo.
echo ============================================================
echo Training completed!
echo ============================================================
pause
