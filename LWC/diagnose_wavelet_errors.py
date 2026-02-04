#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小波变换错误诊断脚本
专门用于重现和诊断小波变换相关的报错
"""

import numpy as np
import traceback
from wavelet_transform import WaveletTransform, calculate_snr, apply_wavelet_denoising

def diagnose_wavelet_errors():
    """诊断小波变换错误"""
    print("=== 小波变换错误诊断 ===\n")
    
    # 测试数据集
    test_cases = [
        ("正常数据", np.random.randn(512)),
        ("含NaN数据", np.array([1, 2, np.nan, 4, 5])),
        ("含无穷大数据", np.array([1, 2, np.inf, 4, 5])),
        ("单点数据", np.array([1.0])),
        ("空数据", np.array([])),
        ("整数数据", np.arange(100)),
        ("光谱模拟数据", np.sin(np.linspace(0, 4*np.pi, 800)) + 0.1*np.random.randn(800))
    ]
    
    for name, data in test_cases:
        print(f"测试案例: {name}")
        print(f"数据形状: {data.shape}, 类型: {data.dtype}")
        
        try:
            # 测试小波变换
            wt = WaveletTransform()
            
            # 分解
            coeffs, lengths = wt.dwt(data, levels=4, extension='sym')
            print(f"✓ DWT成功，系数长度: {len(coeffs)}")
            
            # 重构
            reconstructed = wt.idwt(coeffs, lengths, preserve_length=True)
            print(f"✓ IDWT成功，重构长度: {len(reconstructed)}")
            
            # 去噪
            denoised = apply_wavelet_denoising(data, levels=4)
            print(f"✓ 去噪成功，结果长度: {len(denoised)}")
            
            # 计算SNR
            snr = calculate_snr(data)
            print(f"✓ SNR计算成功: {snr:.2f}")
            
        except Exception as e:
            print(f"✗ 错误: {type(e).__name__}: {e}")
            print(f"  堆栈跟踪:")
            traceback.print_exc()
        
        print("-" * 50)

def test_threshold_calculation():
    """测试阈值计算"""
    print("\n=== 阈值计算测试 ===\n")
    
    wt = WaveletTransform()
    
    # 创建细节系数测试数据
    cD_data = np.random.randn(271)  # 模拟第一层细节系数
    
    try:
        threshold = wt._calculate_threshold(cD_data, level=1)
        print(f"阈值计算成功: {threshold}")
        
        # 验证分组逻辑
        groups = []
        for i in range(0, len(cD_data), 100):
            group = cD_data[i:i+100]
            groups.append(group)
        
        print(f"分组数量: {len(groups)}")
        for i, group in enumerate(groups):
            std_val = np.std(group)
            mean_val = np.mean(group)
            print(f"组{i+1}: 长度={len(group)}, std={std_val:.4f}, mean={mean_val:.4f}")
            
    except Exception as e:
        print(f"阈值计算失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_wavelet_errors()
    test_threshold_calculation()