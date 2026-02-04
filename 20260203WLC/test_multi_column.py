#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试单文件多列数据处理功能
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append('.')

try:
    from corrected_batch_wavelet_snr import BatchWaveletSNRAnalyzer
    
    print("=== 测试单文件多列数据处理 ===")
    
    # 创建测试分析器
    analyzer = BatchWaveletSNRAnalyzer()
    
    # 测试加载多列数据文件
    print("\n1. 测试加载多列数据文件...")
    try:
        filenames = analyzer.load_batch_spectral_files(['multi_column_test.csv'])
        print(f"✓ 成功加载文件")
        print(f"  加载的光谱数量: {len(filenames)}")
        print(f"  光谱名称: {filenames}")
        print(f"  原始数据键: {list(analyzer.original_spectra_data.keys())}")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        sys.exit(1)
    
    # 测试数据插值
    print("\n2. 测试数据插值...")
    try:
        analyzer.interpolate_to_common_grid()
        print(f"✓ 插值完成")
        print(f"  插值矩阵形状: {analyzer.interpolated_original.shape}")
        print(f"  数据范围: {analyzer.interpolated_original.min():.2f} - {analyzer.interpolated_original.max():.2f}")
    except Exception as e:
        print(f"✗ 插值失败: {e}")
        sys.exit(1)
    
    # 测试小波去噪
    print("\n3. 测试小波去噪...")
    try:
        analyzer.batch_wavelet_denoise(wavelet='db4', level=3, threshold_type='soft')
        print(f"✓ 小波去噪完成")
        print(f"  去噪矩阵形状: {analyzer.interpolated_denoised.shape}")
    except Exception as e:
        print(f"✗ 去噪失败: {e}")
        sys.exit(1)
    
    # 测试信噪比计算
    print("\n4. 测试信噪比计算...")
    try:
        snr_results = analyzer.calculate_batch_snr_before_after()
        print(f"✓ 信噪比计算完成")
        print(f"  波长点数: {len(snr_results['wavelength'])}")
        print(f"  处理前平均SNR: {np.mean(snr_results['original_snr']):.2f}")
        print(f"  处理后平均SNR: {np.mean(snr_results['denoised_snr']):.2f}")
    except Exception as e:
        print(f"✗ 信噪比计算失败: {e}")
        sys.exit(1)
    
    print("\n🎉 所有测试通过！单文件多列数据处理功能正常工作。")
    
except ImportError as e:
    print(f"导入模块失败: {e}")
except Exception as e:
    print(f"测试过程中发生错误: {e}")
    import traceback
    traceback.print_exc()