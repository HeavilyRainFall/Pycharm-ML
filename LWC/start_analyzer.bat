@echo off
echo ========================================
echo 光谱数据小波变换分析工具
echo ========================================
echo.

echo 支持文件格式：
echo - CSV文件 (*.csv)
echo - Excel文件 (*.xlsx, *.xls)
echo.

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python环境
    echo 请确保已安装Python 3.7或更高版本
    pause
    exit /b 1
)

REM 检查依赖包
echo 检查依赖包...
python -c "import numpy, pandas, matplotlib, scipy" >nul 2>&1
if %errorlevel% neq 0 (
    echo 安装基础依赖包...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo 基础依赖包安装失败
        pause
        exit /b 1
    )
)

REM 检查Excel支持
echo 检查Excel支持...
python -c "import openpyxl, xlrd" >nul 2>&1
if %errorlevel% neq 0 (
    echo 安装Excel支持包...
    pip install openpyxl xlrd
    if %errorlevel% neq 0 (
        echo Excel支持包安装失败
        pause
        exit /b 1
    )
)

REM 启动GUI程序
echo 启动光谱分析程序...
python spectral_analyzer_gui.py

pause