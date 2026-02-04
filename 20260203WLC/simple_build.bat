@echo off
echo 执行最小化打包...

REM 清理旧文件
echo 清理构建目录...
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

REM 使用PyInstaller进行最小化打包
echo 开始打包...
pyinstaller ^
    --noconfirm ^
    --onedir ^
    --windowed ^
    --name "光谱小波去噪分析系统" ^
    --hidden-import=PyQt5.QtCore ^
    --hidden-import=PyQt5.QtGui ^
    --hidden-import=PyQt5.QtWidgets ^
    --hidden-import=pandas ^
    --hidden-import=numpy ^
    --hidden-import=matplotlib ^
    --hidden-import=matplotlib.backends.backend_qt5agg ^
    --hidden-import=pywt ^
    --exclude-module=tkinter ^
    --exclude-module=unittest ^
    --exclude-module=email ^
    --exclude-module=http ^
    --exclude-module=urllib ^
    --exclude-module=xml ^
    --exclude-module=numpy.random._pickle ^
    --exclude-module=numpy.distutils ^
    --exclude-module=numpy.f2py ^
    --exclude-module=numpy.testing ^
    corrected_batch_wavelet_snr.py

echo 打包完成！
echo 可执行文件在 dist\光谱小波去噪分析系统\ 目录中
pause