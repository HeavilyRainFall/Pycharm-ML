#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试SNR计算问题
"""

import numpy as np
import pandas as pd
from corrected_batch_wavelet_snr import BatchWaveletSNRAnalyzer

def create_test_data():
    """创建测试数据来验证SNR计算"""
    print("=== SNR计算问题诊断 ===\n")
    
    # 创建测试数据：3个文件，每个文件2条光谱
    wavelength = np.linspace(400, 800, 50)
    
    # 文件1：信号较强
    file1_data = pd.DataFrame({
        'Wavelength': wavelength,
        'Spectrum_1': np.sin(wavelength/50) + 2.0 + np.random.normal(0, 0.1, 50),
        'Spectrum_2': np.cos(wavelength/50) + 2.0 + np.random.normal(0, 0.1, 50)
    })
    file1_data.to_csv('snr_test_file1.csv', index=False)
    
    # 文件2：信号中等
    file2_data = pd.DataFrame({
        'Wavelength': wavelength,
        'Spectrum_1': np.sin(wavelength/50) + 1.0 + np.random.normal(0, 0.1, 50),
        'Spectrum_2': np.cos(wavelength/50) + 1.0 + np.random.normal(0, 0.1, 50)
    })
    file2_data.to_csv('snr_test_file2.csv', index=False)
    
    # 文件3：信号较弱
    file3_data = pd.DataFrame({
        'Wavelength': wavelength,
        'Spectrum_1': np.sin(wavelength/50) + 0.5 + np.random.normal(0, 0.1, 50),
        'Spectrum_2': np.cos(wavelength/50) + 0.5 + np.random.normal(0, 0.1, 50)
    })
    file3_data.to_csv('snr_test_file3.csv', index=False)
    
    print("✓ 创建了3个测试文件，信号强度递减")
    return ['snr_test_file1.csv', 'snr_test_file2.csv', 'snr_test_file3.csv']

def test_current_snr_calculation():
    """测试当前的SNR计算方法"""
    print("\n=== 测试当前SNR计算方法 ===")
    
    analyzer = BatchWaveletSNRAnalyzer()
    
    # 加载文件
    file_paths = create_test_data()
    try:
        filenames = analyzer.load_batch_spectral_files(file_paths)
        print(f"✓ 成功加载 {len(filenames)} 个光谱")
        
        # 数据插值
        analyzer.interpolate_to_common_grid()
        print(f"✓ 数据插值完成: {analyzer.interpolated_original.shape}")
        
        # 小波去噪
        analyzer.batch_wavelet_denoise(wavelet='db4', level=3)
        print(f"✓ 小波去噪完成")
        
        # 计算SNR
        snr_results = analyzer.calculate_batch_snr_before_after()
        
        print(f"\n--- 当前SNR计算结果 ---")
        print(f"处理前平均SNR: {np.mean(snr_results['original_snr']):.2f}")
        print(f"处理后平均SNR: {np.mean(snr_results['denoised_snr']):.2f}")
        print(f"SNR改善: {np.mean(snr_results['snr_improvement']):+.2f}")
        
        # 分析问题
        print(f"\n--- 问题分析 ---")
        print(f"原始数据矩阵形状: {analyzer.interpolated_original.shape}")
        print(f"每个波长点的均值范围: {snr_results['original_mean'].min():.3f} - {snr_results['original_mean'].max():.3f}")
        print(f"每个波长点的标准差范围: {snr_results['original_std'].min():.3f} - {snr_results['original_std'].max():.3f}")
        
        # 检查SNR计算是否合理
        reasonable_snr = np.sum(snr_results['original_snr'] < 1000)  # 排除异常大的SNR值
        print(f"合理的SNR值数量: {reasonable_snr}/{len(snr_results['original_snr'])}")
        
        return snr_results
        
    except Exception as e:
        print(f"✗ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def demonstrate_correct_approach():
    """演示正确的SNR计算方法"""
    print("\n=== 正确的SNR计算方法演示 ===")
    
    # 模拟正确的批量SNR计算
    # 对于每个波长点，在所有文件的所有光谱中收集数据
    
    wavelength = np.linspace(400, 800, 50)
    all_signal_values = []  # 每个波长点的所有信号值
    
    # 模拟3个文件，每个文件2条光谱
    for i in range(50):  # 对每个波长点
        wavelength_signals = []
        
        # 文件1的两条光谱
        wavelength_signals.extend([
            np.sin(400/50 + i*(400/50)) + 2.0 + np.random.normal(0, 0.1),
            np.cos(400/50 + i*(400/50)) + 2.0 + np.random.normal(0, 0.1)
        ])
        
        # 文件2的两条光谱
        wavelength_signals.extend([
            np.sin(400/50 + i*(400/50)) + 1.0 + np.random.normal(0, 0.1),
            np.cos(400/50 + i*(400/50)) + 1.0 + np.random.normal(0, 0.1)
        ])
        
        # 文件3的两条光谱
        wavelength_signals.extend([
            np.sin(400/50 + i*(400/50)) + 0.5 + np.random.normal(0, 0.1),
            np.cos(400/50 + i*(400/50)) + 0.5 + np.random.normal(0, 0.1)
        ])
        
        all_signal_values.append(wavelength_signals)
    
    # 计算每个波长点的SNR
    correct_snr_values = []
    for signals in all_signal_values:
        signals_array = np.array(signals)
        mean_val = np.mean(signals_array)
        std_val = np.std(signals_array, ddof=1)
        if std_val > 1e-10:
            snr = mean_val / std_val
        else:
            snr = 1e6
        correct_snr_values.append(snr)
    
    correct_snr_values = np.array(correct_snr_values)
    print(f"✓ 正确方法计算的SNR范围: {correct_snr_values.min():.2f} - {correct_snr_values.max():.2f}")
    print(f"✓ 正确方法平均SNR: {np.mean(correct_snr_values[correct_snr_values < 1000]):.2f}")

def cleanup_test_files():
    """清理测试文件"""
    import os
    test_files = ['snr_test_file1.csv', 'snr_test_file2.csv', 'snr_test_file3.csv']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"✓ 清理测试文件: {file}")

if __name__ == "__main__":
    try:
        # 测试当前方法
        current_results = test_current_snr_calculation()
        
        # 演示正确方法
        demonstrate_correct_approach()
        
        # 清理
        cleanup_test_files()
        
        print("\n=== 诊断结论 ===")
        print("问题根源: 当前SNR计算方法错误地在文件维度计算统计量")
        print("正确方法: 应该在每个波长点收集所有文件的所有光谱数据来计算SNR")
        print("建议: 修改calculate_batch_snr_before_after方法")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()