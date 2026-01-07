# 光谱数据分析程序打包说明

## 打包工具

本项目使用 PyInstaller 将 Python 程序打包为 Windows 可执行文件。

## 打包脚本说明

- `build_exe.py`: 标准打包脚本，包含所有必要的依赖
- `build_minimal_exe.py`: 最小化打包脚本，排除不必要的模块以减小文件大小

## 打包参数说明

### 标准参数
- `--name`: 指定生成的exe文件名
- `--onefile`: 打包成单个exe文件
- `--windowed`: 不显示控制台窗口（适用于GUI应用）
- `--clean`: 清理临时文件
- `--noconfirm`: 覆盖已存在的文件

### 优化参数
- `--exclude-module`: 排除不必要的模块以减小文件大小
- `--hidden-import`: 显式导入PyInstaller可能遗漏的模块
- `--add-data`: 添加额外的数据文件

## 依赖项处理

程序依赖以下主要库：
- pandas: 用于数据处理
- numpy: 用于数值计算
- matplotlib: 用于数据可视化
- scipy: 用于信号处理
- tkinter: 用于GUI界面

## 打包优化策略

1. **排除不必要的模块**：
   - 排除测试模块
   - 排除不使用的功能模块
   - 排除重复的依赖

2. **包含必要的模块**：
   - 显式导入PyInstaller可能遗漏的模块
   - 确保GUI后端正确导入

3. **文件大小优化**：
   - 通过排除模块减少文件大小
   - 权衡功能完整性与文件大小

## 生成的exe文件

- 位置：`dist/` 目录
- 大小：约564MB（包含所有依赖）
- 可在Windows系统上独立运行

## 运行时注意事项

- 首次运行可能需要一些时间加载
- 确保系统有足够的内存
- 部分杀毒软件可能误报，可添加信任

## 重新打包

如需重新打包，请先清理构建文件：
```bash
rmdir /s /q build dist __pycache__ *.spec 2>$null
```

然后运行打包脚本：
```bash
python build_exe.py
```