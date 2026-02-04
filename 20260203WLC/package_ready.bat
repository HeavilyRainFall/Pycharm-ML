@echo off
echo ========================================
echo 光谱小波去噪分析系统 - 打包完成处理
echo ========================================
echo.

REM 等待PyInstaller完成
echo 等待打包完成...
timeout /t 10 /nobreak >nul

REM 检查打包结果
echo 检查打包结果...
if exist "dist\光谱小波去噪分析系统" (
    echo ✓ 打包成功完成！
    echo.
    
    REM 显示文件信息
    echo 打包文件信息:
    echo 路径: %cd%\dist\光谱小波去噪分析系统
    echo.
    
    dir "dist\光谱小波去噪分析系统" /s
    
    echo.
    echo ========================================
    echo 准备分享文件...
    echo ========================================
    
    REM 创建压缩包（如果7-Zip可用）
    if exist "C:\Program Files\7-Zip\7z.exe" (
        echo 使用7-Zip创建压缩包...
        "C:\Program Files\7-Zip\7z.exe" a "光谱小波去噪分析系统.zip" ".\dist\光谱小波去噪分析系统\*"
        if %errorlevel% equ 0 (
            echo ✓ 压缩包创建成功: 光谱小波去噪分析系统.zip
            echo 大小: 
            for %%A in ("光谱小波去噪分析系统.zip") do echo %%~zA 字节
        )
    ) else (
        echo 提示: 安装7-Zip可自动创建压缩包
        echo 当前可直接分享目录: dist\光谱小波去噪分析系统
    )
    
    echo.
    echo ========================================
    echo 分享说明:
    echo ========================================
    echo 1. 将整个 "光谱小波去噪分析系统" 文件夹发送给用户
    echo 2. 用户双击 "光谱小波去噪分析系统.exe" 即可运行
    echo 3. 无需安装Python或其他依赖
    echo 4. 建议连同此说明一起发送
    
) else (
    echo ✗ 打包失败或仍在进行中
    echo 请检查PyInstaller输出或稍后重试
)

echo.
pause