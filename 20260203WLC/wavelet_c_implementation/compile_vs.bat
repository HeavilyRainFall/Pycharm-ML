@echo off
echo 使用Visual Studio编译...

cl /W3 /O2 /Fe:wavelet_test.exe main.c wavelet_transform.c csv_processor.c
cl /W3 /O2 /Fe:wavelet_interactive.exe interactive_main.c wavelet_transform.c csv_processor.c interactive.c

echo 编译完成！
echo 需要在Visual Studio开发者命令提示符中运行此脚本
pause
