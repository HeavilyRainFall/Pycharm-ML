#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小波阈值算法对比测试
验证您提供的算法与当前实现的差异
"""

import numpy as np
from wavelet_transform import WaveletTransform, calculate_snr

def implement_custom_threshold_algorithm(signal, group_size=100):
    """
    实现您描述的小波自动阈值算法
    """
    print("=== 自定义阈值算法实现 ===")
    
    # 1. 小波分解
    wt = WaveletTransform()
    coefficients, lengths = wt.dwt(signal, levels=1)
    
    # 提取第一层细节系数 cD
    # 根据长度信息找到cD的位置
    J = len(lengths) - 2  # 分解层数
    if J >= 1:
        # 第一层细节系数位置
        cD_start = lengths[1]  # 第一层近似系数长度
        cD_end = lengths[1] + lengths[2]  # 第一层细节系数长度
        cD = coefficients[cD_start:cD_end]
        
        print(f"细节系数长度: {len(cD)}")
        
        # 2. 按每group_size个像素分组
        groups = []
        for i in range(0, len(cD), group_size):
            group = cD[i:i+group_size]
            groups.append(group)
        
        print(f"分组数量: {len(groups)}")
        
        # 3. 对每组求std和平均值
        stds = []
        means = []
        for i, group in enumerate(groups):
            std_val = np.std(group)
            mean_val = np.mean(group)
            stds.append(std_val)
            means.append(mean_val)
            print(f"组 {i+1}: std={std_val:.4f}, mean={mean_val:.4f}")
        
        # 4. 计算阈值 t = (1.3 * s̄ / σₛ)^10
        s_bar = np.mean(means)  # 平均值的平均
        sigma_s = np.std(means)  # 平均值的标准差
        
        if sigma_s == 0:
            t = 1000  # 避免除零
        else:
            t = (1.3 * s_bar / sigma_s) ** 10
            
        print(f"s̄ = {s_bar:.4f}, σₛ = {sigma_s:.4f}")
        print(f"计算阈值 t = {t:.4f}")
        
        # 5. 软阈值处理
        cD_denoised = np.zeros_like(cD)
        for i, coeff in enumerate(cD):
            abs_coeff = abs(coeff)
            if abs_coeff < t:
                cD_denoised[i] = 0
            elif coeff > 0:
                cD_denoised[i] = coeff - t
            else:
                cD_denoised[i] = coeff + t
        
        # 6. 重构信号
        # 替换细节系数
        denoised_coefficients = coefficients.copy()
        denoised_coefficients[cD_start:cD_end] = cD_denoised
        
        reconstructed = wt.idwt(denoised_coefficients, lengths, preserve_length=True)
        
        return reconstructed, t
    
    return None, None

def compare_algorithms():
    """对比不同算法的效果"""
    print("=== 算法效果对比 ===\n")
    
    # 创建测试信号（模拟光谱数据）
    x = np.linspace(0, 10, 512)
    signal = np.sin(x * 2 * np.pi) * 100 + np.sin(x * 4 * np.pi) * 50
    noise = np.random.randn(len(signal)) * 20
    noisy_signal = signal + noise
    
    print(f"原始信号长度: {len(noisy_signal)}")
    print(f"信噪比(原始): {calculate_snr(noisy_signal):.2f}")
    
    # 方法1: 当前实现的去噪
    from spectral_analyzer_gui import apply_wavelet_denoising
    try:
        current_denoised = apply_wavelet_denoising(noisy_signal, levels=4, wavelet_name='db4')
        snr_current = calculate_snr(current_denoised)
        print(f"当前去噪后信噪比: {snr_current:.2f}")
    except Exception as e:
        print(f"当前去噪失败: {e}")
        current_denoised = None
    
    # 方法2: 您的自定义算法
    custom_denoised, threshold = implement_custom_threshold_algorithm(noisy_signal)
    if custom_denoised is not None:
        snr_custom = calculate_snr(custom_denoised)
        print(f"自定义算法后信噪比: {snr_custom:.2f}")
        print(f"使用的阈值: {threshold:.4f}")
    
    # 计算差异
    if current_denoised is not None and custom_denoised is not None:
        diff = np.abs(current_denoised - custom_denoised)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"\n两种方法差异:")
        print(f"最大差异: {max_diff:.2f}")
        print(f"平均差异: {mean_diff:.2f}")

if __name__ == "__main__":
    compare_algorithms()