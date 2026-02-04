"""
测试文件加载功能的脚本
用于重现和诊断'ufunc 'isnan' not supported'错误
"""

import pandas as pd
import numpy as np
import os

def test_data_loading():
    """测试数据加载功能"""
    
    # 测试1: 使用示例CSV文件
    print("=== 测试1: CSV文件加载 ===")
    try:
        csv_file = "sample_spectral_data.csv"
        if os.path.exists(csv_file):
            df_csv = pd.read_csv(csv_file)
            print(f"CSV文件读取成功: {df_csv.shape}")
            print(f"列名: {list(df_csv.columns)}")
            print(f"数据类型:\n{df_csv.dtypes}")
            
            # 测试数据转换
            for i in range(1, df_csv.shape[1]):
                counts_series = pd.to_numeric(df_csv.iloc[:, i], errors='coerce')
                counts = counts_series.values
                print(f"第{i+1}列数据类型: {counts.dtype}")
                print(f"NaN数量: {np.isnan(counts).sum()}")
                print(f"无穷大数量: {np.isinf(counts).sum()}")
                
        else:
            print(f"CSV文件不存在: {csv_file}")
    except Exception as e:
        print(f"CSV文件加载错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 使用示例Excel文件
    print("\n=== 测试2: Excel文件加载 ===")
    try:
        excel_file = "sample_spectral_data.xlsx"
        if os.path.exists(excel_file):
            df_excel = pd.read_excel(excel_file)
            print(f"Excel文件读取成功: {df_excel.shape}")
            print(f"列名: {list(df_excel.columns)}")
            print(f"数据类型:\n{df_excel.dtypes}")
            
            # 测试数据转换
            for i in range(1, df_excel.shape[1]):
                counts_series = pd.to_numeric(df_excel.iloc[:, i], errors='coerce')
                counts = counts_series.values
                print(f"第{i+1}列数据类型: {counts.dtype}")
                print(f"NaN数量: {np.isnan(counts).sum()}")
                print(f"无穷大数量: {np.isinf(counts).sum()}")
                
        else:
            print(f"Excel文件不存在: {excel_file}")
    except Exception as e:
        print(f"Excel文件加载错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试3: 模拟有问题的数据
    print("\n=== 测试3: 模拟问题数据 ===")
    try:
        # 创建包含字符串的数据
        test_data = {
            'Wavelength': [400, 401, 402, 403, 404],
            'Counts': [100, 'abc', 200, None, 300]
        }
        df_test = pd.DataFrame(test_data)
        print(f"测试数据:\n{df_test}")
        print(f"数据类型:\n{df_test.dtypes}")
        
        # 尝试转换为数值
        counts_series = pd.to_numeric(df_test['Counts'], errors='coerce')
        counts = counts_series.values
        print(f"转换后数据: {counts}")
        print(f"数据类型: {counts.dtype}")
        
        # 测试isnan
        try:
            mask = ~np.isnan(counts)
            print(f"NaN检查成功: {mask}")
        except Exception as e:
            print(f"NaN检查失败: {e}")
            
    except Exception as e:
        print(f"测试数据错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_loading()