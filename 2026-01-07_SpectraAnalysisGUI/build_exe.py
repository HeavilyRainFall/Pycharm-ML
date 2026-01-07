"""
用于打包光谱数据分析程序为exe文件的脚本
使用方法：python build_exe.py
"""

from PyInstaller.__main__ import run
import sys
import os

# 构建参数 - 优化版本，减少不必要的依赖
args = [
    '--name=光谱数据分析程序',
    '--onefile',  # 打包成单个exe文件
    '--windowed',  # 不显示控制台窗口
    '--clean',  # 清理临时文件
    '--noconfirm',  # 覆盖已存在的文件
    '--collect-data=matplotlib',  # 收集matplotlib数据
    '--collect-data=pandas',  # 收集pandas数据
    '--collect-data=numpy',  # 收集numpy数据
    '--collect-data=scipy',  # 收集scipy数据
    '--collect-data=tkinter',  # 收集tkinter数据
    '--hidden-import=matplotlib.backends.backend_tkagg',  # 隐式导入tkagg后端
    '--hidden-import=scipy.special._ufuncs_cxx',  # 修复scipy导入问题
    '--hidden-import=scipy.special._ufuncs_cxx',  # 修复scipy导入问题
    '--hidden-import=scipy.integrate',  # 修复scipy导入问题
    '--hidden-import=scipy.linalg.cython_blas',  # 修复scipy导入问题
    '--hidden-import=scipy.linalg.cython_lapack',  # 修复scipy导入问题
    '--hidden-import=scipy.spatial.transform.rotation',  # 修复scipy导入问题
    '--hidden-import=scipy.special._ellip_harm_2',  # 修复scipy导入问题
    '--hidden-import=scipy.optimize._highs',  # 修复scipy导入问题
    '--hidden-import=scipy.optimize._highs',  # 修复scipy导入问题
    '--hidden-import=scipy.optimize._highs',  # 修复scipy导入问题
    '--hidden-import=scipy.optimize._highs',  # 修复scipy导入问题
    'spectra_analysis.py'
]

if __name__ == '__main__':
    run(args)