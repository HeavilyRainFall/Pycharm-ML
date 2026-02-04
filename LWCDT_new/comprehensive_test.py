#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立测试文件载入功能
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加当前目录到路径
sys.path.append('.')

def test_detect_header():
    """测试表头检测功能"""
    print("=== 测试表头检测功能 ===")
    
    # 测试无表头文件
    print("\n1. 测试无表头文件:")
    df_no_header = pd.read_csv('test_no_header.csv')
    print("数据预览:")
    print(df_no_header.head())
    
    # 手动实现检测逻辑
    first_row = df_no_header.iloc[0]
    print(f"第一行: {list(first_row)}")
    
    # 检查数值比例
    numeric_count = sum(pd.to_numeric(first_row, errors='coerce').notna())
    print(f"数值元素: {numeric_count}/{len(first_row)}")
    
    has_header = numeric_count < len(first_row) * 0.8
    print(f"判断结果: {'有表头' if has_header else '无表头'}")
    
    # 测试英文表头文件
    print("\n2. 测试英文表头文件:")
    df_en_header = pd.read_csv('test_english_header.csv')
    print("数据预览:")
    print(df_en_header.head())
    
    first_row = df_en_header.iloc[0]
    print(f"第一行: {list(first_row)}")
    
    # 检查关键词
    first_row_str = ' '.join(str(item).lower() for item in first_row)
    header_keywords = ['波长', 'wavelength', 'lambda', 'nm']
    found_keywords = [kw for kw in header_keywords if kw in first_row_str]
    print(f"发现关键词: {found_keywords}")
    
    has_header = bool(found_keywords) or (sum(pd.to_numeric(first_row, errors='coerce').notna()) < len(first_row) * 0.8)
    print(f"判断结果: {'有表头' if has_header else '无表头'}")

def test_load_function():
    """测试载入功能"""
    print("\n=== 测试载入功能 ===")
    
    # 导入载入函数
    try:
        from spectral_wavelet_denoise import load_spectral_data
        print("✓ 成功导入load_spectral_data函数")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return
    
    # 测试各个文件
    test_files = [
        ('test_no_header.csv', '无表头文件'),
        ('test_english_header.csv', '英文表头文件'),
        ('test_chinese_header.csv', '中文表头文件')
    ]
    
    for filename, description in test_files:
        print(f"\n--- 测试 {description}: {filename} ---")
        if os.path.exists(filename):
            try:
                wavelength, spectra, names = load_spectral_data(filename, None, None)
                print(f"✓ 成功载入!")
                print(f"  波长点数: {len(wavelength)}")
                print(f"  光谱数量: {len(spectra)}")
                print(f"  光谱名称: {names}")
                print(f"  波长范围: {wavelength.min():.1f} - {wavelength.max():.1f}")
            except Exception as e:
                print(f"✗ 载入失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"文件不存在: {filename}")

if __name__ == "__main__":
    print("开始文件载入功能测试...\n")
    
    test_detect_header()
    test_load_function()
    
    print("\n=== 测试完成 ===")