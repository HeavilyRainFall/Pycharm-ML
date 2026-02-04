import pandas as pd
import numpy as np

# 直接测试核心功能
print("=== 直接测试文件载入功能 ===")

# 测试1: 无表头文件
print("\n1. 测试无表头文件 test_no_header.csv")
try:
    df = pd.read_csv('test_no_header.csv')
    print("原始数据形状:", df.shape)
    print("前3行:")
    print(df.head(3))
    
    # 检查第一行是否像表头
    first_row = df.iloc[0]
    numeric_count = sum(pd.to_numeric(first_row, errors='coerce').notna())
    print(f"第一行数值比例: {numeric_count}/{len(first_row)} = {numeric_count/len(first_row):.2f}")
    
    if numeric_count >= len(first_row) * 0.8:
        print("判断: 无表头")
        df_final = pd.read_csv('test_no_header.csv', header=None)
    else:
        print("判断: 有表头")
        df_final = df
        
    print("最终数据形状:", df_final.shape)
    print("列名:", list(df_final.columns))
    
except Exception as e:
    print("错误:", e)

# 测试2: 英文表头文件
print("\n2. 测试英文表头文件 test_english_header.csv")
try:
    df = pd.read_csv('test_english_header.csv')
    print("原始数据形状:", df.shape)
    print("前3行:")
    print(df.head(3))
    
    # 检查表头关键词
    first_row_str = ' '.join(str(item).lower() for item in df.iloc[0])
    header_keywords = ['wavelength', '波长']
    found_keywords = [kw for kw in header_keywords if kw in first_row_str]
    print("发现的关键词:", found_keywords)
    
    if found_keywords:
        print("判断: 有表头")
        df_final = df
    else:
        print("判断: 无表头")
        df_final = pd.read_csv('test_english_header.csv', header=None)
        
    print("最终数据形状:", df_final.shape)
    print("列名:", list(df_final.columns))
    
except Exception as e:
    print("错误:", e)