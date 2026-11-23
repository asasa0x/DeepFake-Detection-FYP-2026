@echo off
title Deepfake Detection Model Training
color 0B

echo =======================================================
echo     Starting Deepfake Detection Model Training
echo =======================================================
echo.

cd /d "%~dp0"
if not exist "train.py" (
    echo ERROR: train.py was not found in "%~dp0".
    echo Please add train.py configured to use config.DATASET_DIR ^(Cleaned dataset^) before running training.
    pause
    exit /b 1
)
py train.py --epochs 5 --batch-size 32

echo.
echo =======================================================
echo     Model Training Process Finished.
echo =======================================================
pause
