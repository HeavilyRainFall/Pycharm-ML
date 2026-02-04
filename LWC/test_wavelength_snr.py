#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试波长域信噪比计算功能
"""

import numpy as np
from wavelet_transform import WaveletTransform, apply_wavelet_denoising, calculate_pointwise_snr

def test_wavelength_snr():
    """测试波长域信噪比计算"""
    print("=== 测试波长域信噪比计算 ===\n")
    
    # 生成模拟的多帧光谱数据
    n_frames = 5
    n_wavelengths = 200
    
    print(f"生成 {n_frames} 帧，每帧 {n_wavelengths} 个波长点的模拟数据...")
    
    # 生成基础信号
    wavelength = np.linspace(400, 800, n_wavelengths)
    base_signal = (np.sin(2*np.pi*wavelength/100) * 100 + 
                   np.exp(-((wavelength-600)**2)/2000) * 200 +
                   500)
    
    # 生成多帧数据（带有时间和空间噪声）
    original_data = []
    for i in range(n_frames):
        # 添加时间相关的漂移
        temporal_drift = i * 2
        # 添加空间相关的噪声
        spatial_noise = np.random.normal(0, 10, n_wavelengths)
        frame = base_signal + temporal_drift + spatial_noise
        original_data.append(frame)
    
    original_data = np.array(original_data)
    print(f"原始数据形状: {original_data.shape}")
    
    # 对每帧进行小波去噪
    print("\n对每帧进行小波去噪...")
    denoised_data = []
    for i in range(n_frames):
        denoised_frame = apply_wavelet_denoising(
            original_data[i], 
            levels=4, 
            wavelet_name='db4'
        )
        denoised_data.append(denoised_frame)
    
    denoised_data = np.array(denoised_data)
    print(f"去噪后数据形状: {denoised_data.shape}")
    
    # 计算每个波长点的信噪比
    print("\n计算每个波长点的信噪比...")
    wavelength_snr, wavelength_values = calculate_pointwise_snr(original_data, denoised_data)
    
    print(f"信噪比数组形状: {wavelength_snr.shape}")
    print(f"波长数组形状: {wavelength_values.shape}")
    
    # 显示统计信息
    print(f"\n=== 统计结果 ===")
    print(f"平均信噪比: {np.mean(wavelength_snr):.2f}（比值）")
    print(f"最大信噪比: {np.max(wavelength_snr):.2f}（比值） (波长: {wavelength_values[np.argmax(wavelength_snr)]:.1f} nm)")
    print(f"最小信噪比: {np.min(wavelength_snr):.2f}（比值） (波长: {wavelength_values[np.argmin(wavelength_snr)]:.1f} nm)")
    print(f"信噪比标准差: {np.std(wavelength_snr):.2f}（比值）")
    
    # 显示部分结果
    print(f"\n=== 部分波长点信噪比 ===")
    indices = [0, 50, 100, 150, -1]  # 显示几个代表性点
    for idx in indices:
        actual_idx = idx if idx >= 0 else len(wavelength_snr) + idx
        print(f"波长 {wavelength_values[actual_idx]:.1f} nm: {wavelength_snr[actual_idx]:.2f}（比值）")
    
    # 验证计算合理性
    print(f"\n=== 验证计算合理性 ===")
    # 检查是否有异常值
    valid_snr = wavelength_snr[(wavelength_snr > 0) & (wavelength_snr < 1e6)]
    if len(valid_snr) > 0:
        print(f"✓ 有效信噪比范围: {np.min(valid_snr):.2f} - {np.max(valid_snr):.2f}（比值）")
    else:
        print("⚠ 警告: 没有有效的信噪比值")
    
    # 检查数据一致性
    if np.all(np.isfinite(wavelength_snr)):
        print("✓ 所有信噪比值都是有限数值")
    else:
        print("⚠ 警告: 存在无限或NaN值")
    
    print("\n=== 测试完成 ===")
    return wavelength_snr, wavelength_values

if __name__ == "__main__":
    try:
        snr_values, wavelengths = test_wavelength_snr()
        
        # 可选：绘制结果图
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(wavelengths, snr_values, 'b-', linewidth=2)
            plt.xlabel('波长 (nm)')
            plt.ylabel('信噪比 (比值)')
            plt.title('每个波长点的信噪比')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.hist(snr_values, bins=30, alpha=0.7, color='green', edgecolor='black')
            plt.xlabel('信噪比 (比值)')
            plt.ylabel('频数')
            plt.title('信噪比分布')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("注意: matplotlib未安装，跳过绘图")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()