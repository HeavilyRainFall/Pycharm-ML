import numpy as np
import pandas as pd
import os

# 创建测试数据目录
test_data_dir = "test_spectral_data"
os.makedirs(test_data_dir, exist_ok=True)

# 生成模拟光谱数据
np.random.seed(42)
wavelengths = np.linspace(400, 1000, 200)  # 200个波长点

# 生成5个不同的测试样本
for i in range(5):
    # 基础信号：几个高斯峰
    signal = (np.exp(-((wavelengths - 550)**2)/(2*30**2)) * 0.8 +
              np.exp(-((wavelengths - 700)**2)/(2*25**2)) * 0.6 +
              np.exp(-((wavelengths - 850)**2)/(2*20**2)) * 0.4)
    
    # 添加噪声
    noise_level = 0.1 + 0.05 * i  # 不同噪声水平
    noise = np.random.normal(0, noise_level, len(wavelengths))
    
    # 添加基线漂移
    baseline = 0.1 * np.sin(wavelengths * 0.01) + 0.05 * np.cos(wavelengths * 0.005)
    
    # 合成最终信号
    measured_signal = signal + noise + baseline
    
    # 确保非负
    measured_signal = np.maximum(measured_signal, 0)
    
    # 保存为CSV
    df = pd.DataFrame({
        'Wavelength(nm)': wavelengths,
        'Intensity': measured_signal
    })
    
    filename = f"{test_data_dir}/sample_{i+1}_noisy.csv"
    df.to_csv(filename, index=False)
    print(f"已生成: {filename}")

print(f"\n已在 '{test_data_dir}' 目录下生成5个测试光谱文件")
print("每个文件包含200个波长点的模拟光谱数据，带有不同程度的噪声")