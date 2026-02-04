@echo off
echo ========================================
echo 光谱小波变换去噪分析工具
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python环境
    echo 请先安装Python 3.7或更高版本
    pause
    exit /b 1
)

REM 检查依赖包
echo 检查依赖包...
python -c "import tkinter" >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到tkinter库
    pause
    exit /b 1
)

python -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到numpy库
    echo 请运行: pip install numpy
    pause
    exit /b 1
)

python -c "import matplotlib" >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到matplotlib库
    echo 请运行: pip install matplotlib
    pause
    exit /b 1
)

python -c "import pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到pandas库
    echo 请运行: pip install pandas
    pause
    exit /b 1
)

python -c "import pywt" >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到PyWavelets库
    echo 请运行: pip install PyWavelets
    pause
    exit /b 1
)

echo 依赖包检查通过!
echo.

REM 启动GUI程序
echo 启动光谱小波变换GUI程序...
echo.
python spectral_wavelet_gui.py

pause
