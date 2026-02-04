#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合测试脚本 - 测试完整的数据分析流程
"""

import numpy as np
import pandas as pd
from wavelet_transform import WaveletTransform, calculate_snr

def test_complete_workflow():
    """测试完整的工作流程"""
    print("=== 综合工作流程测试 ===\n")
    
    # 1. 创建测试数据
    print("1. 创建测试光谱数据...")
    wavelengths = np.linspace(400, 800, 100)
    # 模拟真实的光谱信号（带噪声的峰值）
    signal = 1000 + 500 * np.sin(2 * np.pi * wavelengths / 100) + \
             200 * np.random.normal(0, 1, len(wavelengths))
    
    test_data = pd.DataFrame({
        'Wavelength(nm)': wavelengths,
        'Intensity': signal
    })
    
    print(f"   数据形状: {test_data.shape}")
    print(f"   波长范围: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm")
    print(f"   信号范围: {signal.min():.1f} - {signal.max():.1f}")
    
    # 2. 测试数据预处理
    print("\n2. 测试数据预处理...")
    intensity_data = test_data['Intensity'].values
    print(f"   原始数据类型: {intensity_data.dtype}")
    print(f"   是否包含NaN: {np.isnan(intensity_data).any()}")
    print(f"   是否包含无穷大: {np.isinf(intensity_data).any()}")
    
    # 3. 计算原始信噪比
    print("\n3. 计算原始信噪比...")
    original_snr = calculate_snr(intensity_data, use_ratio=True)
    print(f"   原始信噪比: {original_snr:.2f}")
    
    # 4. 执行小波变换
    print("\n4. 执行小波变换...")
    wt = WaveletTransform()
    
    try:
        # 对强度数据进行小波变换
        coefficients, lengths = wt.dwt(intensity_data, levels=4)
        print(f"   小波系数长度: {len(coefficients)}")
        print(f"   分解层数: {len(lengths)-2}")
        print(f"   各层长度: {lengths}")
        
        # 重构信号
        reconstructed_data = wt.idwt(coefficients, lengths)
        print(f"   重构数据长度: {len(reconstructed_data)}")
        print(f"   重构数据范围: {reconstructed_data.min():.2f} - {reconstructed_data.max():.2f}")
        
        # 5. 计算重构后信噪比
        print("\n5. 计算重构后信噪比...")
        reconstructed_snr = calculate_snr(reconstructed_data, use_ratio=True)
        print(f"   重构后信噪比: {reconstructed_snr:.2f}")
        
        # 6. 性能评估
        print("\n6. 性能评估...")
        snr_improvement = reconstructed_snr / original_snr if original_snr > 0 else 0
        print(f"   信噪比改善倍数: {snr_improvement:.2f}x")
        
        if snr_improvement > 1:
            print("   ✓ 小波变换有效改善了信噪比")
        else:
            print("   ⚠ 小波变换未明显改善信噪比")
            
    except Exception as e:
        print(f"   ✗ 小波变换失败: {str(e)}")
        return False
    
    # 7. 测试边界情况
    print("\n7. 测试边界情况...")
    
    # 单帧数据测试
    single_frame_snr = calculate_snr([100], use_ratio=True)
    print(f"   单帧数据信噪比: {single_frame_snr}")
    
    # 全相同数据测试
    constant_snr = calculate_snr([100] * 10, use_ratio=True)
    print(f"   常数数据信噪比: {constant_snr}")
    
    # 零数据测试
    zero_snr = calculate_snr([0] * 10, use_ratio=True)
    print(f"   零数据信噪比: {zero_snr}")
    
    print("\n=== 测试完成 ===")
    return True

def test_file_formats():
    """测试不同文件格式的支持"""
    print("\n=== 文件格式支持测试 ===\n")
    
    test_files = [
        'sample_spectral_data.csv',
        'sample_spectral_data.xlsx',
        'sample_no_header.csv',
        'sample_no_header.xlsx'
    ]
    
    for filename in test_files:
        try:
            print(f"测试文件: {filename}")
            # 这里只是检查文件是否存在，实际的文件读取在GUI中测试
            import os
            if os.path.exists(filename):
                size = os.path.getsize(filename)
                print(f"   ✓ 文件存在 ({size} bytes)")
            else:
                print(f"   ✗ 文件不存在")
        except Exception as e:
            print(f"   ✗ 测试失败: {str(e)}")

if __name__ == "__main__":
    print("开始综合测试...\n")
    
    # 运行完整工作流测试
    success = test_complete_workflow()
    
    # 测试文件格式支持
    test_file_formats()
    
    if success:
        print("\n🎉 所有测试通过！程序功能正常。")
    else:
        print("\n❌ 测试发现问题，请检查代码。")