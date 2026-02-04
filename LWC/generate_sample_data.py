"""
生成示例光谱数据用于测试
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_sample_spectral_data(filename='sample_spectral_data.csv', 
                                n_points=1000, 
                                n_series=3):
    """
    生成示例光谱数据
    
    参数:
    filename: 输出文件名
    n_points: 数据点数
    n_series: 计数序列数量
    """
    
    # 生成波长数据
    wavelength = np.linspace(400, 800, n_points)  # 400-800nm范围
    
    # 生成多个计数序列
    data_dict = {'Wavelength(nm)': wavelength}
    
    for i in range(n_series):
        # 生成基础信号（多个峰）
        signal = np.zeros(n_points)
        
        # 添加几个主要峰
        peaks = [
            (500, 1000, 20),  # 中心波长500nm，强度1000，宽度20
            (550, 800, 15),   # 中心波长550nm，强度800，宽度15
            (650, 1200, 25),  # 中心波长650nm，强度1200，宽度25
        ]
        
        for center, intensity, width in peaks:
            signal += intensity * np.exp(-((wavelength - center)**2)/(2*width**2))
        
        # 添加噪声
        noise_level = intensity * 0.1  # 10%噪声水平
        noisy_signal = signal + np.random.normal(0, noise_level, n_points)
        
        # 确保非负
        noisy_signal = np.maximum(noisy_signal, 0)
        
        # 添加到数据字典
        series_name = f'Counts_{i+1}' if n_series > 1 else 'Counts'
        data_dict[series_name] = noisy_signal
    
    # 创建DataFrame
    df = pd.DataFrame(data_dict)
    
    # 保存到CSV
    df.to_csv(filename, index=False)
    print(f"示例数据已保存到: {filename}")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    return df

def generate_no_header_data(filename='sample_no_header.csv', n_points=500):
    """
    生成无表头的示例数据
    """
    wavelength = np.linspace(300, 700, n_points)
    
    # 生成信号
    signal = 500 * np.exp(-((wavelength - 450)**2)/(2*30**2)) + \
             300 * np.exp(-((wavelength - 550)**2)/(2*20**2)) + \
             700 * np.exp(-((wavelength - 620)**2)/(2*25**2))
    
    # 添加噪声
    noisy_signal = signal + np.random.normal(0, 50, n_points)
    noisy_signal = np.maximum(noisy_signal, 0)
    
    # 创建无表头数据
    data = np.column_stack([wavelength, noisy_signal])
    np.savetxt(filename, data, delimiter=',', fmt='%.6f')
    
    print(f"无表头示例数据已保存到: {filename}")
    return data

def plot_sample_data(df, title="示例光谱数据"):
    """绘制示例数据"""
    plt.figure(figsize=(10, 6))
    
    wavelength = df.iloc[:, 0]
    
    # 绘制所有计数序列
    for i in range(1, df.shape[1]):
        plt.plot(wavelength, df.iloc[:, i], 
                label=df.columns[i], alpha=0.7, linewidth=1)
    
    plt.xlabel('波长 (nm)')
    plt.ylabel('计数值')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 生成带表头的数据
    print("生成带表头的示例数据...")
    df_with_header = generate_sample_spectral_data('sample_spectral_data.csv', 800, 2)
    
    # 生成无表头的数据
    print("\n生成无表头的示例数据...")
    data_no_header = generate_no_header_data('sample_no_header.csv', 600)
    
    # 显示数据预览
    print("\n带表头数据预览:")
    print(df_with_header.head())
    print(f"\n数据统计信息:")
    print(df_with_header.describe())
    
    # 绘制图表
    plot_sample_data(df_with_header, "生成的示例光谱数据")