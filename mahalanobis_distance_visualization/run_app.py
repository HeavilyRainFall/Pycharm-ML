#!/usr/bin/env python
"""
马氏距离与高斯函数关系可视化应用启动脚本
"""

import subprocess
import sys
import os

def check_and_install_dependencies():
    """检查并安装所需的依赖包"""
    required_packages = [
        'streamlit',
        'numpy', 
        'matplotlib',
        'scipy',
        'seaborn'
    ]
    
    print("正在检查依赖包...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"发现缺失的包: {', '.join(missing_packages)}")
        install = input("是否安装这些包? (y/n): ")
        if install.lower() == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                print("依赖包安装完成！")
            except subprocess.CalledProcessError:
                print("尝试直接安装包...")
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
                print("依赖包安装完成！")
        else:
            print("请手动安装所需依赖后重新运行。")
            sys.exit(1)
    else:
        print("所有依赖包已安装。")

def run_streamlit_app():
    """运行Streamlit应用"""
    print("正在启动马氏距离可视化应用...")
    print("请在浏览器中打开 http://localhost:8501 查看应用")
    
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动应用时出错: {e}")
    except KeyboardInterrupt:
        print("\n应用已停止。")

if __name__ == "__main__":
    # 切换到脚本所在目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("马氏距离与高斯函数关系可视化应用")
    print("="*50)
    
    check_and_install_dependencies()
    run_streamlit_app()