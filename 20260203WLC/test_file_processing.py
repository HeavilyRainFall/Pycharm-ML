#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试文件处理逻辑的脚本
"""

import pandas as pd
import numpy as np
import os
from corrected_batch_wavelet_snr import BatchWaveletSNRAnalyzer

def create_test_files():
    """创建测试文件"""
    # 创建两列文件（标准格式）
    two_col_data = pd.DataFrame({
        'Wavelength': np.linspace(400, 800, 100),
        'Intensity': np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
    })
    two_col_data.to_csv('test_two_columns.csv', index=False)
    print("✓ 创建测试文件: test_two_columns.csv (2列)")
    
    # 创建三列文件（多列格式）
    three_col_data = pd.DataFrame({
        'Wavelength': np.linspace(400, 800, 100),
        'Sample_A': np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100),
        'Sample_B': np.cos(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100)
    })
    three_col_data.to_csv('test_three_columns.csv', index=False)
    print("✓ 创建测试文件: test_three_columns.csv (3列)")

def test_processing_logic():
    """测试处理逻辑"""
    analyzer = BatchWaveletSNRAnalyzer()
    
    print("\n=== 测试两列文件处理 ===")
    try:
        result = analyzer.load_batch_spectral_files(['test_two_columns.csv'])
        print(f"处理结果: {len(result)} 个光谱")
        print(f"光谱名称: {result}")
    except Exception as e:
        print(f"处理失败: {e}")
    
    print("\n=== 测试三列文件处理 ===")
    try:
        result = analyzer.load_batch_spectral_files(['test_three_columns.csv'])
        print(f"处理结果: {len(result)} 个光谱")
        print(f"光谱名称: {result}")
    except Exception as e:
        print(f"处理失败: {e}")

def cleanup_test_files():
    """清理测试文件"""
    test_files = ['test_two_columns.csv', 'test_three_columns.csv']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ 清理测试文件: {file}")

if __name__ == "__main__":
    print("开始测试文件处理逻辑...")
    create_test_files()
    test_processing_logic()
    cleanup_test_files()
    print("测试完成!")