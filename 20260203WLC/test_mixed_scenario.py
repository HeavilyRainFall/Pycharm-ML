#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试多文件混合场景
"""

import pandas as pd
import numpy as np
import os
from corrected_batch_wavelet_snr import BatchWaveletSNRAnalyzer

def create_mixed_test_files():
    """创建混合测试文件"""
    # 创建多个两列文件
    for i in range(3):
        data = pd.DataFrame({
            'Wavelength': np.linspace(400, 800, 100),
            'Intensity': np.sin(np.linspace(0, 4*np.pi, 100) + i) + np.random.normal(0, 0.1, 100)
        })
        filename = f'multi_two_col_{i+1}.csv'
        data.to_csv(filename, index=False)
        print(f"✓ 创建两列文件: {filename}")
    
    # 创建一个多列文件
    multi_data = pd.DataFrame({
        'Wavelength': np.linspace(400, 800, 100),
        'Sample_X': np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100),
        'Sample_Y': np.cos(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.1, 100),
        'Sample_Z': np.tan(np.linspace(0, 2*np.pi, 100)) + np.random.normal(0, 0.1, 100)
    })
    multi_data.to_csv('multi_column_file.csv', index=False)
    print("✓ 创建多列文件: multi_column_file.csv (4列)")

def test_mixed_scenario():
    """测试混合场景"""
    analyzer = BatchWaveletSNRAnalyzer()
    
    # 准备所有测试文件
    test_files = [
        'multi_two_col_1.csv',
        'multi_two_col_2.csv', 
        'multi_two_col_3.csv',
        'multi_column_file.csv'
    ]
    
    print(f"\n=== 测试混合场景处理 ({len(test_files)} 个文件) ===")
    try:
        result = analyzer.load_batch_spectral_files(test_files)
        print(f"处理结果: {len(result)} 个光谱")
        print("光谱详情:")
        for i, spec_name in enumerate(result, 1):
            print(f"  {i}. {spec_name}")
    except Exception as e:
        print(f"处理失败: {e}")

def cleanup_test_files():
    """清理测试文件"""
    test_files = [
        'multi_two_col_1.csv',
        'multi_two_col_2.csv', 
        'multi_two_col_3.csv',
        'multi_column_file.csv'
    ]
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ 清理测试文件: {file}")

if __name__ == "__main__":
    print("开始测试混合场景处理...")
    create_mixed_test_files()
    test_mixed_scenario()
    cleanup_test_files()
    print("混合场景测试完成!")