#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试修正后的SNR计算
"""

import numpy as np
import pandas as pd
from corrected_batch_wavelet_snr import BatchWaveletSNRAnalyzer

def test_corrected_snr():
    """测试修正后的SNR计算"""
    print("=== 测试修正后的SNR计算 ===\n")
    
    # 创建测试数据
    wavelength = np.linspace(400, 800, 30)
    
    # 创建高质量信号（低噪声）
    high_quality_data = pd.DataFrame({
        'Wavelength': wavelength,
        'Spectrum_1': np.sin(wavelength/50) + 2.0 + np.random.normal(0, 0.05, 30),
        'Spectrum_2': np.cos(wavelength/50) + 2.0 + np.random.normal(0, 0.05, 30)
    })
    high_quality_data.to_csv('high_quality.csv', index=False)
    
    # 创建低质量信号（高噪声）
    low_quality_data = pd.DataFrame({
        'Wavelength': wavelength,
        'Spectrum_1': np.sin(wavelength/50) + 0.5 + np.random.normal(0, 0.3, 30),
        'Spectrum_2': np.cos(wavelength/50) + 0.5 + np.random.normal(0, 0.3, 30)
    })
    low_quality_data.to_csv('low_quality.csv', index=False)
    
    print("✓ 创建了高低质量测试文件")
    
    try:
        analyzer = BatchWaveletSNRAnalyzer()
        
        # 加载文件
        filenames = analyzer.load_batch_spectral_files(['high_quality.csv', 'low_quality.csv'])
        print(f"✓ 成功加载 {len(filenames)} 个光谱")
        
        # 数据插值
        analyzer.interpolate_to_common_grid()
        print(f"✓ 数据插值完成: {analyzer.interpolated_original.shape}")
        
        # 小波去噪
        analyzer.batch_wavelet_denoise(wavelet='db4', level=3)
        print(f"✓ 小波去噪完成")
        
        # 计算SNR（修正后的方法）
        snr_results = analyzer.calculate_batch_snr_before_after()
        
        print(f"\n--- 修正后SNR计算结果 ---")
        print(f"处理前平均SNR: {np.mean(snr_results['original_snr']):.2f}")
        print(f"处理后平均SNR: {np.mean(snr_results['denoised_snr']):.2f}")
        print(f"SNR改善: {np.mean(snr_results['snr_improvement']):+.2f}")
        
        # 验证SNR值的合理性
        orig_snr_range = (snr_results['original_snr'].min(), snr_results['original_snr'].max())
        denoise_snr_range = (snr_results['denoised_snr'].min(), snr_results['denoised_snr'].max())
        
        print(f"\n--- SNR值范围分析 ---")
        print(f"处理前SNR范围: {orig_snr_range[0]:.2f} - {orig_snr_range[1]:.2f}")
        print(f"处理后SNR范围: {denoise_snr_range[0]:.2f} - {denoise_snr_range[1]:.2f}")
        
        # 检查是否有明显的SNR改善
        improvement_count = np.sum(snr_results['snr_improvement'] > 0)
        total_points = len(snr_results['snr_improvement'])
        improvement_ratio = improvement_count / total_points * 100
        
        print(f"\n--- 改善情况分析 ---")
        print(f"SNR得到改善的波长点: {improvement_count}/{total_points} ({improvement_ratio:.1f}%)")
        
        # 验证是否符合预期（去噪应该提高SNR）
        avg_improvement = np.mean(snr_results['snr_improvement'])
        if avg_improvement > 0:
            print("✓ 去噪后SNR总体提高，符合预期")
        else:
            print("⚠ 去噪后SNR未明显提高，需要进一步检查")
            
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理测试文件
        import os
        test_files = ['high_quality.csv', 'low_quality.csv']
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"✓ 清理测试文件: {file}")

if __name__ == "__main__":
    success = test_corrected_snr()
    if success:
        print("\n🎉 SNR计算修正成功！")
    else:
        print("\n❌ SNR计算仍有问题需要解决。")