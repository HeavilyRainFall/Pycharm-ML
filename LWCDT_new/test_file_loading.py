#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文件载入功能
"""

from spectral_wavelet_denoise import load_spectral_data
import os
import pandas as pd

def test_file_loading():
    """测试不同格式文件的载入"""
    print("=== 文件载入功能测试 ===\n")
    
    # 测试文件列表
    test_files = [
        'test_no_header.csv',
        'test_english_header.csv', 
        'test_chinese_header.csv'
    ]
    
    for filename in test_files:
        print(f"--- 测试文件: {filename} ---")
        if os.path.exists(filename):
            try:
                # 显示文件内容预览
                print("文件内容预览:")
                df_preview = pd.read_csv(filename, nrows=3)
                print(df_preview.to_string())
                print()
                
                # 测试载入功能
                wavelength, spectra, names = load_spectral_data(filename, None, None)
                print(f"✓ 成功载入!")
                print(f"  波长数据长度: {len(wavelength)}")
                print(f"  光谱数量: {len(spectra)}")
                print(f"  光谱名称: {names}")
                print(f"  波长范围: {wavelength.min():.2f} - {wavelength.max():.2f}")
                print()
                
            except Exception as e:
                print(f"✗ 载入失败: {e}")
                print()
        else:
            print(f"文件不存在: {filename}\n")

if __name__ == "__main__":
    test_file_loading()