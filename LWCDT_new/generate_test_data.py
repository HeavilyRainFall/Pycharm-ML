"""
生成测试光谱数据用于GUI程序测试
"""

import pandas as pd
import numpy as np
import os

def generate_test_spectral_data():
    """生成多种测试数据文件"""
    
    # 创建输出目录
    output_dir = "test_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("生成测试光谱数据...")
    
    # 1. 生成单个文件测试数据
    print("1. 生成单个文件数据...")
    
    # 波长范围
    wavelength = np.linspace(400, 800, 512)
    
    # 生成多条光谱（模拟不同采集条件）
    spectra_data = {
        'spectrum_1': (np.exp(-(wavelength-500)**2/50) + 
                      0.8*np.exp(-(wavelength-600)**2/80) + 
                      0.6*np.exp(-(wavelength-700)**2/60)),
        'spectrum_2': (0.9*np.exp(-(wavelength-480)**2/45) + 
                      1.2*np.exp(-(wavelength-580)**2/70) + 
                      0.7*np.exp(-(wavelength-680)**2/55)),
        'spectrum_3': (1.1*np.exp(-(wavelength-520)**2/55) + 
                      0.7*np.exp(-(wavelength-620)**2/75) + 
                      0.9*np.exp(-(wavelength-720)**2/65))
    }
    
    # 添加不同类型噪声
    noise_levels = [0.05, 0.1, 0.15]
    
    for i, (name, spectrum) in enumerate(spectra_data.items()):
        # 添加噪声
        noisy_spectrum = spectrum + np.random.normal(0, noise_levels[i], len(wavelength))
        # 确保非负
        noisy_spectrum = np.maximum(noisy_spectrum, 0)
        spectra_data[name] = noisy_spectrum
    
    # 保存单个文件（包含多条光谱）
    df_single = pd.DataFrame({'wavelength': wavelength})
    for name, spectrum in spectra_data.items():
        df_single[name] = spectrum
    
    df_single.to_csv(os.path.join(output_dir, 'single_file_test.csv'), index=False)
    print(f"  已保存: single_file_test.csv (包含3条光谱)")
    
    # 2. 生成多个独立文件
    print("2. 生成多个独立文件...")
    
    for i, (name, spectrum) in enumerate(spectra_data.items()):
        # 为每个光谱创建独立文件
        df_individual = pd.DataFrame({
            'wavelength': wavelength,
            f'{name}_data': spectrum
        })
        
        filename = f'individual_{i+1}_{name}.csv'
        df_individual.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"  已保存: {filename}")
    
    # 3. 生成Excel格式文件
    print("3. 生成Excel格式文件...")
    
    df_excel = pd.DataFrame({'wavelength': wavelength})
    for name, spectrum in spectra_data.items():
        df_excel[name] = spectrum
    
    df_excel.to_excel(os.path.join(output_dir, 'excel_format_test.xlsx'), index=False)
    print(f"  已保存: excel_format_test.xlsx")
    
    # 4. 生成带有表头的文件
    print("4. 生成带表头文件...")
    
    df_with_headers = pd.DataFrame({
        'Wavelength(nm)': wavelength,
        'Sample_A': spectra_data['spectrum_1'],
        'Sample_B': spectra_data['spectrum_2'],
        'Sample_C': spectra_data['spectrum_3']
    })
    
    df_with_headers.to_csv(os.path.join(output_dir, 'header_format_test.csv'), index=False)
    print(f"  已保存: header_format_test.csv")
    
    print(f"\n测试数据生成完成！")
    print(f"数据已保存至目录: {output_dir}")
    print(f"包含以下文件:")
    for filename in os.listdir(output_dir):
        if filename.endswith(('.csv', '.xlsx')):
            print(f"  - {filename}")
    
    return output_dir

def generate_batch_test_folder():
    """生成批量处理测试文件夹"""
    batch_dir = "batch_test_folder"
    if not os.path.exists(batch_dir):
        os.makedirs(batch_dir)
    
    print(f"\n生成批量处理测试文件夹: {batch_dir}")
    
    # 生成10个不同的光谱文件用于批量测试
    wavelength = np.linspace(400, 800, 256)
    
    for i in range(10):
        # 生成不同的光谱模式
        center_wavelength = 500 + i * 20  # 不同中心波长
        amplitude = 0.5 + np.random.random() * 1.0  # 随机振幅
        width = 30 + np.random.random() * 40  # 随机宽度
        
        spectrum = amplitude * np.exp(-(wavelength-center_wavelength)**2/width)
        
        # 添加噪声
        noise = np.random.normal(0, 0.1, len(wavelength))
        noisy_spectrum = spectrum + noise
        noisy_spectrum = np.maximum(noisy_spectrum, 0)
        
        # 保存文件
        df = pd.DataFrame({
            'wavelength': wavelength,
            f'spectrum_{i+1:02d}': noisy_spectrum
        })
        
        filename = f'batch_sample_{i+1:02d}.csv'
        df.to_csv(os.path.join(batch_dir, filename), index=False)
    
    print(f"批量测试文件夹生成完成，包含10个CSV文件")
    
    return batch_dir

if __name__ == "__main__":
    # 生成所有测试数据
    test_dir = generate_test_spectral_data()
    batch_dir = generate_batch_test_folder()
    
    print(f"\n" + "="*50)
    print("测试数据准备就绪！")
    print("="*50)
    print("您可以使用以下数据测试GUI程序:")
    print(f"1. 单文件测试: {test_dir}/single_file_test.csv")
    print(f"2. 批量处理测试: {batch_dir}/")
    print(f"3. Excel格式测试: {test_dir}/excel_format_test.xlsx")
    print(f"4. 表头格式测试: {test_dir}/header_format_test.csv")
