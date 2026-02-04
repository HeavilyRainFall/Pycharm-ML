#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试虚拟拆分多列数据方案
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加当前目录到路径
sys.path.append('.')

try:
    from corrected_batch_wavelet_snr import BatchWaveletSNRAnalyzer
    
    print("=== 测试虚拟拆分多列数据方案 ===")
    
    # 创建测试分析器
    analyzer = BatchWaveletSNRAnalyzer()
    
    # 创建测试数据文件
    print("\n1. 创建测试数据文件...")
    
    # 带表头的多列数据
    df_with_header = pd.DataFrame({
        'Wavelength(nm)': np.linspace(400, 500, 20),
        'Sample_A': np.sin(np.linspace(0, 4*np.pi, 20)) * 100 + 1000,
        'Sample_B': np.cos(np.linspace(0, 4*np.pi, 20)) * 80 + 950,
        'Sample_C': np.sin(np.linspace(0, 2*np.pi, 20)) * 120 + 1020
    })
    df_with_header.to_csv('virtual_test_with_header.csv', index=False)
    print("  ✓ 创建带表头测试文件: virtual_test_with_header.csv")
    
    # 无表头的多列数据
    df_without_header = pd.DataFrame({
        'col1': np.linspace(400, 500, 15),
        'col2': np.random.normal(100, 10, 15),
        'col3': np.random.normal(95, 8, 15),
        'col4': np.random.normal(105, 12, 15)
    })
    df_without_header.to_csv('virtual_test_without_header.csv', index=False, header=False)
    print("  ✓ 创建无表头测试文件: virtual_test_without_header.csv")
    
    # 测试虚拟拆分加载
    print("\n2. 测试虚拟拆分加载...")
    try:
        filenames = analyzer.load_batch_spectral_files([
            'virtual_test_with_header.csv', 
            'virtual_test_without_header.csv'
        ])
        print(f"✓ 虚拟拆分加载成功")
        print(f"  总虚拟光谱数: {len(filenames)}")
        print(f"  虚拟光谱名称: {filenames}")
        print(f"  原始数据字典大小: {len(analyzer.original_spectra_data)}")
    except Exception as e:
        print(f"✗ 虚拟拆分加载失败: {e}")
        raise
    
    # 验证数据结构
    print("\n3. 验证数据结构...")
    for name, (wavelength, intensity) in list(analyzer.original_spectra_data.items())[:3]:
        print(f"  {name}:")
        print(f"    波长范围: {wavelength.min():.1f} - {wavelength.max():.1f} nm")
        print(f"    数据范围: {intensity.min():.2f} - {intensity.max():.2f}")
        print(f"    数据点数: {len(intensity)}")
    
    # 测试处理流程
    print("\n4. 测试完整处理流程...")
    try:
        # 数据插值
        analyzer.interpolate_to_common_grid()
        print(f"  ✓ 数据插值完成: {analyzer.interpolated_original.shape}")
        
        # 小波去噪
        analyzer.batch_wavelet_denoise(wavelet='db4', level=3)
        print(f"  ✓ 小波去噪完成: {analyzer.interpolated_denoised.shape}")
        
        # 信噪比计算
        snr_results = analyzer.calculate_batch_snr_before_after()
        print(f"  ✓ 信噪比计算完成")
        print(f"    波长点数: {len(snr_results['wavelength'])}")
        print(f"    处理前平均SNR: {np.mean(snr_results['original_snr']):.2f}")
        print(f"    处理后平均SNR: {np.mean(snr_results['denoised_snr']):.2f}")
        
    except Exception as e:
        print(f"✗ 处理流程失败: {e}")
        raise
    
    print("\n🎉 虚拟拆分方案测试通过！")
    print("\n方案优势:")
    print("✓ 无需生成实际物理文件")
    print("✓ 内存中高效处理多列数据")
    print("✓ 保持原有接口兼容性")
    print("✓ 支持有表头和无表头文件")
    print("✓ 清晰的虚拟文件命名")
    print("✓ 所有光谱享有同等处理权重")
    
except ImportError as e:
    print(f"导入模块失败: {e}")
except Exception as e:
    print(f"测试过程中发生错误: {e}")
    import traceback
    traceback.print_exc()
finally:
    # 清理测试文件
    test_files = ['virtual_test_with_header.csv', 'virtual_test_without_header.csv']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"已清理测试文件: {file}")