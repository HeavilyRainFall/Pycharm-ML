@echo off
echo ========================================
echo 光谱小波去噪分析系统 - PyInstaller打包
echo ========================================
echo.

REM 检查Python环境
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python环境
    pause
    exit /b 1
)

REM 检查PyInstaller
echo 检查PyInstaller...
python -c "import PyInstaller" >nul 2>&1
if %errorlevel% neq 0 (
    echo 安装PyInstaller...
    pip install pyinstaller
    if %errorlevel% neq 0 (
        echo PyInstaller安装失败
        pause
        exit /b 1
    )
)

REM 清理之前的构建文件
echo 清理之前的构建文件...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"
if exist "*.spec" del "*.spec.bak"

echo.

REM 执行打包
echo 开始打包...
pyinstaller wavelet_snr.spec

if %errorlevel% equ 0 (
    echo.
    echo ========================================
    echo 打包成功！
    echo ========================================
    echo 可执行文件位置: dist\光谱小波去噪分析系统.exe
    echo 文件大小: 
    if exist "dist\光谱小波去噪分析系统.exe" (
        for %%A in ("dist\光谱小波去噪分析系统.exe") do echo %%~zA 字节
    )
    echo.
    echo 如需测试运行，请双击 dist\光谱小波去噪分析系统.exe
) else (
    echo.
    echo ========================================
    echo 打包失败！
    echo ========================================
    echo 请检查错误信息并重试
)

echo.
pause