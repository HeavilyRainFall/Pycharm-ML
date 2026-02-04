#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证小波变换长度一致性修复
"""

import numpy as np
from wavelet_transform import WaveletTransform

def test_length_consistency_fix():
    """测试长度一致性修复"""
    print("=== 长度一致性修复验证 ===\n")
    
    test_lengths = [100, 128, 200, 256, 512]
    
    for original_length in test_lengths:
        print(f"测试长度: {original_length}")
        
        # 生成测试信号
        signal = np.sin(np.linspace(0, 4*np.pi, original_length)) + 0.1*np.random.randn(original_length)
        
        # 执行小波变换
        wt = WaveletTransform()
        
        try:
            # 分解
            coefficients, lengths = wt.dwt(signal, levels=4)
            
            # 重构（保持长度）
            reconstructed_preserve = wt.idwt(coefficients, lengths, preserve_length=True)
            
            # 重构（不保持长度）
            reconstructed_full = wt.idwt(coefficients, lengths, preserve_length=False)
            
            print(f"  原始长度: {original_length}")
            print(f"  重构(保持长度): {len(reconstructed_preserve)}")
            print(f"  重构(完整长度): {len(reconstructed_full)}")
            print(f"  长度差异: {len(reconstructed_full) - len(reconstructed_preserve)}")
            
            # 验证数据质量
            if len(reconstructed_preserve) == original_length:
                diff = np.abs(signal - reconstructed_preserve)
                max_error = np.max(diff)
                print(f"  ✓ 长度保持一致，最大误差: {max_error:.2e}")
            else:
                print(f"  ✗ 长度仍有问题")
                
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
        
        print("-" * 40)

def demonstrate_snr_methods():
    """演示不同的信噪比计算方法"""
    print("\n=== 信噪比计算方法演示 ===\n")
    
    # 生成模拟的多帧数据
    n_frames = 10
    frame_length = 100
    
    print("1. 帧间信噪比计算（当前方法）:")
    frames = []
    for i in range(n_frames):
        # 模拟略有不同的帧（加入时间噪声）
        base_signal = np.sin(np.linspace(0, 4*np.pi, frame_length)) * 100
        noise = np.random.randn(frame_length) * 5
        frame = base_signal + noise + i * 0.5  # 加入轻微漂移
        frames.append(frame)
    
    frames_array = np.array(frames)
    
    # 帧间信噪比计算
    from wavelet_transform import calculate_snr
    snr_interframe = calculate_snr(frames_array.flatten(), use_ratio=True)
    print(f"   多帧数据信噪比: {snr_interframe:.2f}")
    
    # 单帧信噪比比较
    print("\n2. 单帧信噪比比较:")
    snr_single_frames = []
    for i, frame in enumerate(frames[:3]):  # 只显示前3帧
        snr_single = calculate_snr(frame, use_ratio=True)
        snr_single_frames.append(snr_single)
        print(f"   帧 {i+1} 信噪比: {snr_single:.2f}")
    
    print(f"   平均单帧信噪比: {np.mean(snr_single_frames):.2f}")
    print(f"   帧间信噪比: {snr_interframe:.2f}")
    
    print("\n结论:")
    print("- 帧间信噪比反映了时间稳定性")
    print("- 单帧信噪比反映了单次测量质量")
    print("- 两者结合可以全面评估系统性能")

if __name__ == "__main__":
    test_length_consistency_fix()
    demonstrate_snr_methods()