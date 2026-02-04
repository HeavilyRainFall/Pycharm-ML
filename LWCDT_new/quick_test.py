#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试优化后的功能
"""

import sys
import os
sys.path.append('.')

def quick_test():
    """快速测试文件载入和去噪功能"""
    print("=== 快速功能测试 ===\n")
    
    try:
        from spectral_wavelet_denoise import load_spectral_data, SpectralWaveletDenoiser
        import numpy as np
        print("✓ 成功导入所需模块")
        
        # 测试文件载入
        print("\n1. 测试文件载入功能:")
        test_file = 'test_no_header.csv'
        if os.path.exists(test_file):
            wavelength, spectra, names = load_spectral_data(test_file, None, None)
            print(f"   ✓ 成功载入 {len(spectra)} 条光谱")
            print(f"   ✓ 波长范围: {wavelength.min():.1f} - {wavelength.max():.1f}")
        else:
            print("   ✗ 测试文件不存在")
            return
            
        # 测试小波去噪
        print("\n2. 测试小波去噪功能:")
        denoiser = SpectralWaveletDenoiser(wavelet='db4', level=6)
        
        # 对第一条光谱进行去噪
        original_spectrum = spectra[0]
        denoised_spectrum = denoiser.denoise_single_spectrum(original_spectrum)
        
        print(f"   ✓ 去噪处理完成")
        print(f"   ✓ 原始数据长度: {len(original_spectrum)}")
        print(f"   ✓ 去噪数据长度: {len(denoised_spectrum)}")
        
        # 简单的信噪比计算
        print("\n3. 简单信噪比分析:")
        noise = original_spectrum - denoised_spectrum
        signal_mean = np.mean(np.abs(denoised_spectrum))
        noise_std = np.std(noise)
        
        if noise_std > 0:
            snr = signal_mean / noise_std
            print(f"   ✓ 估算信噪比: {snr:.2f}")
        else:
            print(f"   ✓ 噪声标准差为零，信噪比无法计算")
            
        print("\n=== 测试完成 ===")
        print("所有功能正常工作！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_test()