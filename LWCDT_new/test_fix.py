#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的解包问题
"""

import sys
import os
sys.path.append('.')

def test_denoise_return_values():
    """测试去噪函数的返回值"""
    print("=== 测试去噪函数返回值 ===\n")
    
    try:
        from spectral_wavelet_denoise import SpectralWaveletDenoiser
        from spectral_wavelet_gui import SpectralWaveletGUI
        import numpy as np
        
        print("✓ 成功导入模块")
        
        # 测试SpectralWaveletDenoiser类
        print("\n1. 测试SpectralWaveletDenoiser类:")
        denoiser = SpectralWaveletDenoiser(wavelet='db4', level=6)
        
        # 生成测试数据
        test_spectrum = np.random.normal(0, 1, 100) + np.sin(np.linspace(0, 4*np.pi, 100))
        
        # 测试单条光谱去噪
        denoised_spectrum, threshold = denoiser.denoise_single_spectrum(test_spectrum)
        print(f"   ✓ denoise_single_spectrum返回两个值")
        print(f"   ✓ 去噪后数据长度: {len(denoised_spectrum)}")
        print(f"   ✓ 使用的阈值: {threshold:.6f}")
        
        # 测试批量去噪
        test_spectra = np.array([test_spectrum, test_spectrum * 1.2, test_spectrum * 0.8])
        denoised_spectra, thresholds = denoiser.batch_denoise(test_spectra)
        print(f"   ✓ batch_denoise返回两个值")
        print(f"   ✓ 去噪后矩阵形状: {denoised_spectra.shape}")
        print(f"   ✓ 阈值列表长度: {len(thresholds)}")
        
        # 测试GUI中的调用方式
        print("\n2. 测试GUI中的调用方式:")
        gui = SpectralWaveletGUI(None)  # 创建GUI实例（不需要实际运行）
        
        # 模拟GUI中的调用
        denoised_spectrum, threshold = gui.denoise_single_spectrum(test_spectrum, denoiser)
        print(f"   ✓ GUI中的denoise_single_spectrum返回两个值")
        print(f"   ✓ 去噪后数据长度: {len(denoised_spectrum)}")
        print(f"   ✓ 使用的阈值: {threshold:.6f}")
        
        print("\n=== 测试完成，所有功能正常！===")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_denoise_return_values()
    if success:
        print("\n🎉 修复成功！解包问题已解决。")
    else:
        print("\n❌ 仍有问题需要修复。")