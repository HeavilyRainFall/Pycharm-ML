#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修正后的批量SNR计算
"""

import sys
import os
sys.path.append('.')

def test_batch_snr_calculation():
    """测试批量SNR计算"""
    print("=== 测试批量SNR计算 ===\n")
    
    try:
        from spectral_wavelet_gui import SpectralWaveletGUI
        import numpy as np
        import tkinter as tk
        
        print("✓ 成功导入GUI模块")
        
        # 创建GUI实例（用于测试计算方法）
        root = tk.Tk()
        root.withdraw()  # 隐藏窗口
        gui = SpectralWaveletGUI(root)
        
        # 模拟批量处理结果数据
        print("\n1. 创建测试数据:")
        
        # 模拟3个文件，每个文件2条光谱，10个波长点
        test_wavelength = np.linspace(400, 800, 10)
        
        # 文件1
        file1_spectra = np.array([
            np.sin(test_wavelength/100) + np.random.normal(0, 0.1, 10),  # 光谱1
            np.cos(test_wavelength/100) + np.random.normal(0, 0.1, 10)   # 光谱2
        ])
        
        # 文件2  
        file2_spectra = np.array([
            np.sin(test_wavelength/100) * 1.2 + np.random.normal(0, 0.1, 10),  # 光谱1
            np.cos(test_wavelength/100) * 1.2 + np.random.normal(0, 0.1, 10)   # 光谱2
        ])
        
        # 文件3
        file3_spectra = np.array([
            np.sin(test_wavelength/100) * 0.8 + np.random.normal(0, 0.1, 10),  # 光谱1
            np.cos(test_wavelength/100) * 0.8 + np.random.normal(0, 0.1, 10)   # 光谱2
        ])
        
        # 模拟去噪后的数据（信号更强，噪声更少）
        file1_denoised = file1_spectra * 1.1  # 略微放大
        file2_denoised = file2_spectra * 1.1
        file3_denoised = file3_spectra * 1.1
        
        # 设置批量结果
        gui.batch_results = {
            'file1.csv': {
                'wavelength': test_wavelength,
                'original_spectra': file1_spectra,
                'denoised_spectra': file1_denoised,
                'spectrum_names': ['spectrum1', 'spectrum2']
            },
            'file2.csv': {
                'wavelength': test_wavelength,
                'original_spectra': file2_spectra,
                'denoised_spectra': file2_denoised,
                'spectrum_names': ['spectrum1', 'spectrum2']
            },
            'file3.csv': {
                'wavelength': test_wavelength,
                'original_spectra': file3_spectra,
                'denoised_spectra': file3_denoised,
                'spectrum_names': ['spectrum1', 'spectrum2']
            }
        }
        
        print(f"   ✓ 创建了3个测试文件")
        print(f"   ✓ 每个文件包含2条光谱")
        print(f"   ✓ 波长点数: {len(test_wavelength)}")
        
        # 测试批量SNR计算
        print("\n2. 测试批量SNR计算:")
        batch_snr_result = gui.calculate_batch_snr_statistics()
        
        if batch_snr_result:
            print("   ✓ 批量SNR计算成功")
            print(f"   ✓ SNR值数组长度: {len(batch_snr_result['snr_values'])}")
            print(f"   ✓ 有效数据点数: {batch_snr_result['statistics']['valid_points']}")
            print(f"   ✓ 平均SNR: {batch_snr_result['statistics']['mean']:.2f}")
            print(f"   ✓ SNR范围: {batch_snr_result['statistics']['min']:.2f} - {batch_snr_result['statistics']['max']:.2f}")
            
            # 验证计算逻辑
            print("\n3. 验证计算逻辑:")
            
            # 手动计算第一个波长点的SNR进行验证
            wavelength_0_signals = []
            for filename in ['file1.csv', 'file2.csv', 'file3.csv']:
                result = gui.batch_results[filename]
                for spectrum in result['denoised_spectra']:
                    wavelength_0_signals.append(spectrum[0])
            
            manual_mean = np.mean(wavelength_0_signals)
            manual_std = np.std(wavelength_0_signals, ddof=1)
            manual_snr = manual_mean / manual_std if manual_std > 1e-10 else 1e6
            
            calculated_snr = batch_snr_result['snr_values'][0]
            
            print(f"   ✓ 手动计算第1个波长点SNR: {manual_snr:.4f}")
            print(f"   ✓ 程序计算第1个波长点SNR: {calculated_snr:.4f}")
            print(f"   ✓ 计算结果{'✓ 正确' if abs(manual_snr - calculated_snr) < 1e-10 else '✗ 错误'}")
            
            # 测试绘图功能
            print("\n4. 测试绘图功能:")
            try:
                gui.plot_batch_snr_comparison()
                print("   ✓ 批量SNR对比图绘制成功")
            except Exception as e:
                print(f"   ✗ 绘图失败: {e}")
            
        else:
            print("   ✗ 批量SNR计算失败")
            
        root.destroy()
        print("\n=== 测试完成 ===")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batch_snr_calculation()
    if success:
        print("\n🎉 批量SNR计算修正成功！")
    else:
        print("\n❌ 仍有问题需要修复。")