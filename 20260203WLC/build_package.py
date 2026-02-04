#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光谱小波去噪分析系统 - 最小化PyInstaller打包脚本
"""

import subprocess
import sys
import os
import shutil

def check_dependencies():
    """检查必要的依赖"""
    print("检查依赖...")
    
    # 检查PyInstaller
    try:
        import PyInstaller
        print("✓ PyInstaller 已安装")
    except ImportError:
        print("✗ PyInstaller 未安装，正在安装...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller 安装完成")
    
    # 检查主要依赖
    required_packages = ['PyQt5', 'pandas', 'numpy', 'matplotlib', 'PyWavelets']
    for package in required_packages:
        try:
            __import__(package.replace('PyWavelets', 'pywt'))
            print(f"✓ {package} 已安装")
        except ImportError:
            print(f"✗ {package} 未安装，正在安装...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 安装完成")

def clean_previous_build():
    """清理之前的构建文件"""
    print("\n清理之前的构建文件...")
    
    dirs_to_remove = ['build', 'dist']
    files_to_remove = [f for f in os.listdir('.') if f.endswith('.spec')]
    
    for directory in dirs_to_remove:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"✓ 删除目录: {directory}")
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ 删除文件: {file}")

def run_pyinstaller():
    """执行PyInstaller打包"""
    print("\n开始PyInstaller打包...")
    
    # 构建命令
    cmd = [
        'pyinstaller',
        '--noconfirm',           # 不询问确认
        '--onedir',              # 生成目录形式而非单文件
        '--windowed',            # 无控制台窗口
        '--name', '光谱小波去噪分析系统',  # 应用名称
        # 必需的隐藏导入
        '--hidden-import=PyQt5.QtCore',
        '--hidden-import=PyQt5.QtGui',
        '--hidden-import=PyQt5.QtWidgets',
        '--hidden-import=pandas',
        '--hidden-import=numpy',
        '--hidden-import=matplotlib',
        '--hidden-import=matplotlib.backends.backend_qt5agg',
        '--hidden-import=pywt',
        # 排除不必要的模块
        '--exclude-module=tkinter',
        '--exclude-module=unittest',
        '--exclude-module=email',
        '--exclude-module=http',
        '--exclude-module=urllib',
        '--exclude-module=xml',
        '--exclude-module=numpy.random._pickle',
        '--exclude-module=numpy.distutils',
        '--exclude-module=numpy.f2py',
        '--exclude-module=numpy.testing',
        # 主程序文件
        'corrected_batch_wavelet_snr.py'
    ]
    
    print("执行命令:", ' '.join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✓ 打包成功完成!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("✗ 打包失败!")
        print("错误输出:", e.stderr)
        print("标准输出:", e.stdout)
        return False

def show_results():
    """显示打包结果"""
    print("\n" + "="*50)
    print("打包结果:")
    print("="*50)
    
    exe_path = os.path.join('dist', '光谱小波去噪分析系统', '光谱小波去噪分析系统.exe')
    
    if os.path.exists(exe_path):
        size = os.path.getsize(exe_path)
        print(f"✓ 可执行文件: {exe_path}")
        print(f"✓ 文件大小: {size:,} 字节 ({size/1024/1024:.1f} MB)")
        print(f"✓ 所在目录: {os.path.dirname(exe_path)}")
        print("\n🎉 打包成功！")
        print("   双击可执行文件即可运行程序")
    else:
        print("✗ 未找到可执行文件")
        print("   请检查打包过程中的错误信息")

def main():
    """主函数"""
    print("="*60)
    print("光谱小波去噪分析系统 - PyInstaller最小化打包")
    print("="*60)
    
    try:
        # 检查依赖
        check_dependencies()
        
        # 清理旧文件
        clean_previous_build()
        
        # 执行打包
        if run_pyinstaller():
            # 显示结果
            show_results()
        else:
            print("\n❌ 打包过程中出现错误，请检查上述信息")
            
    except Exception as e:
        print(f"\n❌ 打包过程出现异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()