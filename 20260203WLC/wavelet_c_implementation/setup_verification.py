#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C语言小波变换程序验证和编译指导脚本
"""

import os
import subprocess
import sys

def check_compilers():
    """检查可用的C编译器"""
    compilers = []
    
    # 检查GCC
    try:
        result = subprocess.run(['gcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            compilers.append(('GCC', 'gcc'))
            print("✓ GCC编译器可用")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # 检查Clang
    try:
        result = subprocess.run(['clang', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            compilers.append(('Clang', 'clang'))
            print("✓ Clang编译器可用")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # 检查MSVC (Windows)
    try:
        result = subprocess.run(['cl'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0 and 'Microsoft' in result.stderr:  # cl存在但需要环境设置
            compilers.append(('MSVC', 'cl'))
            print("✓ MSVC编译器可用（需要开发者命令提示符）")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    return compilers

def generate_compile_scripts(compilers):
    """为不同编译器生成编译脚本"""
    
    # Bash/Linux编译脚本
    bash_script = """#!/bin/bash
# Linux/MacOS编译脚本

echo "编译C语言小波变换程序..."

# 基本测试程序
gcc -Wall -O2 -std=c99 -lm -o wavelet_test main.c wavelet_transform.c csv_processor.c

# 交互式程序
gcc -Wall -O2 -std=c99 -lm -o wavelet_interactive interactive_main.c wavelet_transform.c csv_processor.c interactive.c

echo "编译完成！"
echo "运行测试: ./wavelet_test"
echo "交互式使用: ./wavelet_interactive"
"""
    
    # Windows批处理脚本
    bat_script = """@echo off
echo 编译C语言小波变换程序...

REM 检查MinGW GCC
gcc --version >nul 2>&1
if %errorlevel% equ 0 (
    echo 使用GCC编译...
    gcc -Wall -O2 -std=c99 -lm -o wavelet_test.exe main.c wavelet_transform.c csv_processor.c
    gcc -Wall -O2 -std=c99 -lm -o wavelet_interactive.exe interactive_main.c wavelet_transform.c csv_processor.c interactive.c
    echo 编译完成！
    echo 运行测试: wavelet_test.exe
    echo 交互式使用: wavelet_interactive.exe
    goto end
)

REM 检查Clang
clang --version >nul 2>&1
if %errorlevel% equ 0 (
    echo 使用Clang编译...
    clang -Wall -O2 -std=c99 -lm -o wavelet_test.exe main.c wavelet_transform.c csv_processor.c
    clang -Wall -O2 -std=c99 -lm -o wavelet_interactive.exe interactive_main.c wavelet_transform.c csv_processor.c interactive.c
    echo 编译完成！
    goto end
)

echo 未找到合适的C编译器
echo 请安装以下任一编译器：
echo 1. MinGW-w64 (推荐)
echo 2. Visual Studio Build Tools
echo 3. Clang for Windows

:end
pause
"""
    
    # Visual Studio开发者命令提示符脚本
    vs_bat_script = """@echo off
echo 使用Visual Studio编译...

cl /W3 /O2 /Fe:wavelet_test.exe main.c wavelet_transform.c csv_processor.c
cl /W3 /O2 /Fe:wavelet_interactive.exe interactive_main.c wavelet_transform.c csv_processor.c interactive.c

echo 编译完成！
echo 需要在Visual Studio开发者命令提示符中运行此脚本
pause
"""

    # 保存脚本文件
    with open('compile_linux.sh', 'w', encoding='utf-8') as f:
        f.write(bash_script)
    
    with open('compile_windows.bat', 'w', encoding='utf-8') as f:
        f.write(bat_script)
        
    with open('compile_vs.bat', 'w', encoding='utf-8') as f:
        f.write(vs_bat_script)
    
    # 设置Linux脚本可执行权限
    try:
        os.chmod('compile_linux.sh', 0o755)
    except:
        pass
    
    print("✓ 已生成编译脚本:")
    print("  - compile_linux.sh (Linux/MacOS)")
    print("  - compile_windows.bat (Windows MinGW)")
    print("  - compile_vs.bat (Visual Studio)")

def create_test_data():
    """创建测试数据文件"""
    import numpy as np
    
    # 生成测试信号
    x = np.linspace(0, 10, 256)
    signal = (np.sin(x) + 0.5 * np.sin(3*x) + 0.3 * np.cos(5*x) + 
              np.random.normal(0, 0.1, 256))
    
    # 保存为CSV格式
    with open('test_signal.csv', 'w') as f:
        f.write('Index,Wavelength,Intensity\n')
        for i, val in enumerate(signal):
            f.write(f'{i},{400 + i * 1.5625},{val}\n')
    
    print("✓ 已创建测试数据文件: test_signal.csv")

def main():
    print("=== C语言小波变换程序验证 ===\n")
    
    # 检查编译器
    print("1. 检查可用编译器:")
    compilers = check_compilers()
    
    if not compilers:
        print("⚠ 未检测到C编译器")
        print("\n请安装以下任一编译器:")
        print("• MinGW-w64 (Windows): https://www.mingw-w64.org/")
        print("• Visual Studio Build Tools (Windows)")
        print("• GCC (Linux/MacOS)")
        print("• Clang (跨平台)")
    else:
        print(f"\n✓ 检测到 {len(compilers)} 个编译器")
        for name, cmd in compilers:
            print(f"  - {name}: {cmd}")
    
    # 生成编译脚本
    print("\n2. 生成编译脚本...")
    generate_compile_scripts(compilers)
    
    # 创建测试数据
    print("\n3. 创建测试数据...")
    create_test_data()
    
    print("\n=== 使用说明 ===")
    print("1. 根据您的系统选择合适的编译脚本运行")
    print("2. 编译成功后运行:")
    print("   • 基本测试: ./wavelet_test (Linux) 或 wavelet_test.exe (Windows)")
    print("   • 交互式: ./wavelet_interactive (Linux) 或 wavelet_interactive.exe (Windows)")
    print("3. 交互式命令:")
    print("   • load test_signal.csv 2    # 加载第2列数据")
    print("   • process 4 0.1             # 4层分解，阈值0.1")
    print("   • snr                       # 计算信噪比")
    print("   • save result.csv           # 保存结果")
    
    print("\n程序特点:")
    print("✓ 完整实现DB4小波变换")
    print("✓ 支持软/硬阈值去噪")
    print("✓ CSV文件读写功能")
    print("✓ 交互式命令行界面")
    print("✓ 信噪比计算和验证")

if __name__ == "__main__":
    main()