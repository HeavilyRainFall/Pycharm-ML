# 光谱数据分析程序打包指南

本指南介绍如何将光谱数据分析程序打包成独立的exe文件。

## 打包前准备

1. 确保已安装所有依赖包：
```bash
pip install -r requirements.txt
```

2. 确保PyInstaller已安装：
```bash
pip install pyinstaller
```

## 打包命令

### 方法1：使用构建脚本
```bash
python build_exe.py
```

### 方法2：直接使用PyInstaller命令
```bash
pyinstaller --name=光谱数据分析程序 --onefile --windowed --add-data="README.md;." --add-data="requirements.txt;." spectra_analysis.py
```

## 参数说明

- `--name`: 指定生成的exe文件名
- `--onefile`: 打包成单个exe文件
- `--windowed`: 不显示控制台窗口（适用于GUI应用）
- `--add-data`: 添加额外文件到exe中
- `spectra_analysis.py`: 要打包的主程序文件

## 输出位置

打包完成后，exe文件将位于 `dist/` 目录下。

## 注意事项

1. 打包过程可能需要几分钟时间，请耐心等待
2. 生成的exe文件可能较大（通常几十MB到上百MB）
3. 打包后的exe文件可以在没有安装Python环境的Windows系统上运行
4. 如果需要自定义图标，可添加 `--icon=path/to/icon.ico` 参数