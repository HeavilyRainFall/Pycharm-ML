#!/bin/bash
# Linux/MacOS编译脚本

echo "编译C语言小波变换程序..."

# 基本测试程序
gcc -Wall -O2 -std=c99 -lm -o wavelet_test main.c wavelet_transform.c csv_processor.c

# 交互式程序
gcc -Wall -O2 -std=c99 -lm -o wavelet_interactive interactive_main.c wavelet_transform.c csv_processor.c interactive.c

echo "编译完成！"
echo "运行测试: ./wavelet_test"
echo "交互式使用: ./wavelet_interactive"
