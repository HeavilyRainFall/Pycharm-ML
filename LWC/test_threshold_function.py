#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试阈值计算函数
"""

import numpy as np
from wavelet_transform import WaveletTransform

def test_threshold_calculation():
    """直接测试阈值计算"""
    print("=== 直接测试阈值计算 ===\n")
    
    # 创建测试数据
    test_data = np.random.randn(271)  # 模拟细节系数长度
    
    print(f"测试数据长度: {len(test_data)}")
    
    # 创建小波变换对象
    wt = WaveletTransform()
    
    try:
        # 测试阈值计算
        threshold = wt._calculate_threshold(test_data, level=1)
        print(f"计算的阈值: {threshold:.4f}")
        
        # 验证分组逻辑
        groups = []
        for i in range(0, len(test_data), 100):
            group = test_data[i:i+100]
            groups.append(group)
        
        print(f"分组数量: {len(groups)}")
        for i, group in enumerate(groups):
            std_val = np.std(group)
            mean_val = np.mean(group)
            print(f"组{i+1}: 长度={len(group)}, std={std_val:.4f}, mean={mean_val:.4f}")
        
        # 计算理论阈值
        means = [np.mean(g) for g in groups]
        s_bar = np.mean(means)
        sigma_s = np.std(means)
        t_theory = (1.3 * s_bar / sigma_s) ** 10 if sigma_s != 0 else 1000
        if t_theory > 1000:
            t_theory = 1000
            
        print(f"s̄ = {s_bar:.4f}, σₛ = {sigma_s:.4f}")
        print(f"理论阈值: {t_theory:.4f}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_threshold_calculation()