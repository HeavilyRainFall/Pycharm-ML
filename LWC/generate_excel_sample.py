"""
生成Excel格式的示例光谱数据用于测试
"""

import pandas as pd
import numpy as np

def generate_excel_sample_data():
    """生成Excel示例数据"""
    
    # 生成波长数据
    wavelength = np.linspace(400, 800, 500)
    
    # 生成计数数据
    counts1 = 500 * np.exp(-((wavelength - 500)**2)/(2*30**2)) + \
              300 * np.exp(-((wavelength - 600)**2)/(2*25**2)) + \
              700 * np.exp(-((wavelength - 700)**2)/(2*35**2))
    
    counts2 = 400 * np.exp(-((wavelength - 480)**2)/(2*28**2)) + \
              600 * np.exp(-((wavelength - 580)**2)/(2*22**2)) + \
              800 * np.exp(-((wavelength - 680)**2)/(2*30**2))
    
    # 添加噪声
    noise1 = np.random.normal(0, 30, len(wavelength))
    noise2 = np.random.normal(0, 25, len(wavelength))
    
    counts1 = np.maximum(counts1 + noise1, 0)
    counts2 = np.maximum(counts2 + noise2, 0)
    
    # 创建带表头的数据
    df_with_header = pd.DataFrame({
        'Wavelength(nm)': wavelength,
        'Counts_1': counts1,
        'Counts_2': counts2
    })
    
    # 创建无表头的数据
    df_no_header = pd.DataFrame(np.column_stack([wavelength, counts1]))
    
    # 保存为Excel文件
    with pd.ExcelWriter('sample_spectral_data.xlsx', engine='openpyxl') as writer:
        df_with_header.to_excel(writer, sheet_name='Sheet1', index=False)
    
    with pd.ExcelWriter('sample_no_header.xlsx', engine='openpyxl') as writer:
        df_no_header.to_excel(writer, sheet_name='Data', index=False, header=False)
    
    print("Excel示例文件已生成:")
    print("- sample_spectral_data.xlsx (带表头)")
    print("- sample_no_header.xlsx (无表头)")
    print(f"数据点数: {len(wavelength)}")
    
    return df_with_header, df_no_header

if __name__ == "__main__":
    df1, df2 = generate_excel_sample_data()
    print("\n带表头数据预览:")
    print(df1.head())
    print("\n无表头数据预览:")
    print(df2.head())