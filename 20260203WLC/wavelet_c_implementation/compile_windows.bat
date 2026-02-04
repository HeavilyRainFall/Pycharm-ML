@echo off
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
