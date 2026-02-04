"""
用于打包最小化光谱数据分析程序为exe文件的脚本
使用方法：python build_minimal_exe.py
"""

import PyInstaller.__main__
import os

# 构建参数 - 最小化版本，仅包含必要的依赖
args = [
    'spectra_analysis.py',
    '--name=光谱数据分析程序（信噪比、动态范围、分辨率）',
    '--onefile',  # 打包成单个exe文件
    '--windowed',  # 不显示控制台窗口
    '--clean',  # 清理临时文件
    '--noconfirm',  # 覆盖已存在的文件
    '--exclude-module=tkinter.test',  # 排除测试模块
    '--exclude-module=tcl',  # 排除tcl相关模块（如果不需要高级GUI功能）
    '--exclude-module=PIL',  # 排除PIL（如果matplotlib有内置支持）
    '--exclude-module=IPython',  # 排除IPython
    '--exclude-module=matplotlib.tests',  # 排除matplotlib测试
    '--exclude-module=numpy.random._pickle',  # 排除不必要的模块
    '--exclude-module=scipy.special._ufuncs_cxx',  # 排除可能引起问题的模块
    '--exclude-module=scipy.integrate',  # 排除不需要的scipy模块
    '--exclude-module=scipy.linalg.cython_blas',  # 排除不需要的模块
    '--exclude-module=scipy.linalg.cython_lapack',  # 排除不需要的模块
    '--exclude-module=scipy.spatial.transform.rotation',  # 排除不需要的模块
    '--exclude-module=scipy.special._ellip_harm_2',  # 排除不需要的模块
    '--exclude-module=scipy.optimize._highs',  # 排除不需要的模块
    '--hidden-import=matplotlib.backends.backend_tkagg',  # 隐式导入tkagg后端
    '--hidden-import=tkinter',  # 隐式导入tkinter
    '--hidden-import=matplotlib.pyplot',  # 隐式导入pyplot
    '--hidden-import=numpy.core.multiarray',  # 隐式导入numpy核心模块
    '--add-data=README.md;.',  # 添加README文件
    '--add-data=requirements.txt;.',  # 添加requirements文件
]

if __name__ == '__main__':
    PyInstaller.__main__.run(args)