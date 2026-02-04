#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小波变换数据长度变化分析
"""

import numpy as np
from wavelet_transform import WaveletTransform

def analyze_data_length_consistency():
    """分析小波变换前后数据长度一致性"""
    print("=== 小波变换数据长度分析 ===\n")
    
    # 创建测试数据
    test_lengths = [100, 128, 200, 256, 512, 1000]
    
    for original_length in test_lengths:
        print(f"原始数据长度: {original_length}")
        
        # 生成测试信号
        signal = np.random.randn(original_length)
        
        # 执行小波变换
        wt = WaveletTransform()
        
        try:
            # 分解
            coefficients, lengths = wt.dwt(signal, levels=4)
            print(f"  小波系数长度: {len(coefficients)}")
            print(f"  各层长度: {lengths}")
            
            # 重构
            reconstructed = wt.idwt(coefficients, lengths)
            reconstructed_length = len(reconstructed)
            
            print(f"  重构后长度: {reconstructed_length}")
            print(f"  长度变化: {reconstructed_length - original_length}")
            
            # 检查数据一致性
            if reconstructed_length == original_length:
                print(f"  ✓ 长度保持一致")
            else:
                print(f"  ⚠ 长度发生变化")
                
            # 检查数值精度
            diff = np.abs(signal - reconstructed[:original_length])
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            print(f"  最大误差: {max_diff:.2e}")
            print(f"  平均误差: {mean_diff:.2e}")
            
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
        
        print("-" * 40)

def analyze_snr_calculation_method():
    """分析信噪比计算方法"""
    print("\n=== 信噪比计算方法分析 ===\n")
    
    print("当前实现的信噪比计算方法:")
    print("1. 时间域信噪比（帧间信噪比）:")
    print("   - 输入: 多帧光谱数据序列")
    print("   - 计算: 相邻帧之间的差异作为噪声估计")
    print("   - 公式: SNR = signal_power / noise_power")
    print("   - 其中: noise_power = var(diff(signal_frames))")
    print()
    
    print("2. 频域信噪比（小波域信噪比）:")
    print("   - 输入: 单帧光谱数据")
    print("   - 计算: 小波系数的能量分布")
    print("   - 公式: SNR = approximation_energy / detail_energy")
    print()
    
    print("当前程序采用的是时间域信噪比计算方法，适用于:")
    print("- 多帧连续采集的光谱数据")
    print("- 评估系统的时间稳定性")
    print("- 检测帧间噪声水平")
    print()
    
    print("如果您需要频域信噪比或其他计算方法，请告诉我具体需求!")

if __name__ == "__main__":
    analyze_data_length_consistency()
    analyze_snr_calculation_method()