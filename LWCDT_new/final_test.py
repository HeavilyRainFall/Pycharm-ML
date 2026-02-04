#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件载入功能测试
"""

from spectral_wavelet_denoise import load_spectral_data
import os

print('=== 测试文件载入功能 ===')

# 测试文件列表
test_files = ['test_no_header.csv', 'test_english_header.csv', 'test_chinese_header.csv']

for filename in test_files:
    print(f'\n--- 测试文件: {filename} ---')
    if os.path.exists(filename):
        try:
            wavelength, spectra, names = load_spectral_data(filename, None, None)
            print(f'✓ 成功载入!')
            print(f'  波长数据长度: {len(wavelength)}')
            print(f'  光谱数量: {len(spectra)}')
            print(f'  光谱名称: {names}')
            print(f'  波长范围: {wavelength.min():.1f} - {wavelength.max():.1f}')
        except Exception as e:
            print(f'✗ 载入失败: {e}')
            import traceback
            traceback.print_exc()
    else:
        print(f'文件不存在: {filename}')