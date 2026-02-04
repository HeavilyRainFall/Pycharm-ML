#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试核心去噪功能
"""

import sys
import os
sys.path.append('.')

def simple_test():
    """简单测试去噪功能"""
    print("=== 简单功能测试 ===\n")
    
    try:
        from spectral_wavelet_denoise import SpectralWaveletDenoiser
        import numpy as np
        
        print("✓ 成功导入去噪器")
        
        # 创建去噪器
        denoiser = SpectralWaveletDenoiser(wavelet='db4', level=6)
        print("✓ 成功创建去噪器实例")
        
        # 生成测试数据
        test_spectrum = np.random.normal(0, 0.5, 100) + np.sin(np.linspace(0, 4*np.pi, 100))
        print("✓ 成功生成测试数据")
        
        # 测试去噪（关键测试）
        denoised_spectrum, threshold = denoiser.denoise_single_spectrum(test_spectrum)
        print("✓ 成功执行去噪，返回两个值")
        print(f"✓ 阈值: {threshold:.6f}")
        print(f"✓ 原始数据长度: {len(test_spectrum)}")
        print(f"✓ 去噪数据长度: {len(denoised_spectrum)}")
        
        print("\n🎉 核心功能测试通过！解包问题已解决。")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    simple_test()