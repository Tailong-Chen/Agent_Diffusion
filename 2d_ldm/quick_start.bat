@echo off
REM 快速启动脚本 - LDM训练 (Windows版本)
REM 使用前请修改TIFF_PATH为您的数据路径

setlocal enabledelayedexpansion

echo ==================================================
echo   TIFF堆栈LDM训练 - 快速启动
echo ==================================================

REM ============================================
REM 配置区域 - 请修改这里
REM ============================================

REM 您的TIFF文件路径（必须修改）
set "TIFF_PATH=.\data\your_data.tif"

REM 输出目录
set "OUTPUT_DIR=.\output_ldm"

REM 图像尺寸 (512 或 1024)
set "IMAGE_SIZE=1024"

REM 批次大小 (1024用2, 512用6)
set "BATCH_SIZE=2"

REM 是否只训练AutoEncoder (true/false)
set "ONLY_AE=false"

REM 是否只训练Diffusion (true/false)
set "ONLY_DIFF=false"

REM AutoEncoder checkpoint路径 (如果ONLY_DIFF=true)
set "AE_CHECKPOINT="

REM ============================================
REM 检查Python
REM ============================================

echo.
echo 1️⃣  检查Python环境...

python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安装或不在PATH中
    echo 请安装Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python已安装

REM ============================================
REM 检查依赖
REM ============================================

echo.
echo 2️⃣  检查依赖包...

python -c "import torch" 2>nul
if errorlevel 1 (
    echo ❌ PyTorch未安装
    echo 请运行: pip install -r requirements_tiff_ldm.txt
    pause
    exit /b 1
)

python -c "import monai" 2>nul
if errorlevel 1 (
    echo ❌ MONAI未安装
    echo 请运行: pip install -r requirements_tiff_ldm.txt
    pause
    exit /b 1
)

echo ✅ 依赖检查通过

REM ============================================
REM 检查TIFF文件
REM ============================================

echo.
echo 3️⃣  检查TIFF数据...

if not exist "%TIFF_PATH%" (
    echo ❌ TIFF文件不存在: %TIFF_PATH%
    echo 请修改脚本中的TIFF_PATH变量
    pause
    exit /b 1
)

echo ✅ TIFF文件存在: %TIFF_PATH%
echo.

REM 运行数据检查
python check_tiff_data.py --tiff_path "%TIFF_PATH%"

echo.
set /p CONTINUE="是否继续训练? (y/n): "
if /i not "%CONTINUE%"=="y" (
    echo 训练已取消
    pause
    exit /b 0
)

REM ============================================
REM 开始训练
REM ============================================

echo.
echo 4️⃣  开始训练...
echo.

REM 构建训练命令
set "TRAIN_CMD=python train_tiff_ldm.py --tiff_path "%TIFF_PATH%" --output_dir "%OUTPUT_DIR%" --image_size %IMAGE_SIZE% --batch_size %BATCH_SIZE%"

REM 添加可选参数
if "%ONLY_AE%"=="true" (
    set "TRAIN_CMD=!TRAIN_CMD! --skip_diffusion"
    echo 📝 模式: 仅训练AutoEncoder
)

if "%ONLY_DIFF%"=="true" (
    set "TRAIN_CMD=!TRAIN_CMD! --skip_autoencoder"
    if not "%AE_CHECKPOINT%"=="" (
        set "TRAIN_CMD=!TRAIN_CMD! --autoencoder_checkpoint "%AE_CHECKPOINT%""
    )
    echo 📝 模式: 仅训练Diffusion
)

echo 📝 训练配置:
echo    - TIFF文件: %TIFF_PATH%
echo    - 输出目录: %OUTPUT_DIR%
echo    - 图像尺寸: %IMAGE_SIZE%×%IMAGE_SIZE%
echo    - 批次大小: %BATCH_SIZE%
echo.
echo 🚀 执行命令:
echo %TRAIN_CMD%
echo.

REM 执行训练
%TRAIN_CMD%

REM ============================================
REM 训练完成
REM ============================================

echo.
echo ==================================================
echo   🎉 训练完成！
echo ==================================================
echo.
echo 📁 输出文件:
echo    - Checkpoints: %OUTPUT_DIR%\checkpoints\
echo    - 样本图像: %OUTPUT_DIR%\samples\
echo    - 训练曲线: %OUTPUT_DIR%\training_history.png
echo.
echo 🎨 生成新样本:
echo python generate_samples.py ^
echo     --checkpoint %OUTPUT_DIR%\checkpoints\diffusion_epoch_250.pth ^
echo     --num_samples 20 ^
echo     --output_dir .\generated
echo.
echo ==================================================

pause

