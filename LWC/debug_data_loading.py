#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据加载调试脚本
专门用于诊断和解决数据加载问题
"""

import pandas as pd
import numpy as np
import os

def debug_data_loading(file_path):
    """详细调试数据加载过程"""
    print(f"=== 调试文件: {file_path} ===")
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"✗ 文件不存在: {file_path}")
            return False
            
        print(f"✓ 文件存在，大小: {os.path.getsize(file_path)} bytes")
        
        # 根据文件扩展名选择读取方法
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.xlsx', '.xls']:
            # Excel文件处理
            try:
                print(f"\n处理Excel文件...")
                df = pd.read_excel(file_path)
                print(f"✓ 成功读取Excel文件")
                print(f"数据形状: {df.shape}")
                print(f"列名: {list(df.columns)}")
                print(f"数据类型:\n{df.dtypes}")
                print(f"前5行数据:")
                print(df.head())
                full_df = df
            except Exception as e:
                print(f"✗ Excel读取失败: {e}")
                return False
        else:
            # CSV文件处理
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            
            for encoding in encodings:
                try:
                    print(f"\n尝试编码: {encoding}")
                    
                    # 先读取前几行进行分析
                    sample_df = pd.read_csv(file_path, nrows=10, encoding=encoding)
                    print(f"✓ 成功读取前10行")
                    print(f"数据形状: {sample_df.shape}")
                    print(f"列名: {list(sample_df.columns)}")
                    print(f"数据类型:\n{sample_df.dtypes}")
                    print(f"前5行数据:")
                    print(sample_df.head())
                    
                    # 读取完整文件
                    full_df = pd.read_csv(file_path, encoding=encoding)
                    print(f"✓ 成功读取完整文件，形状: {full_df.shape}")
                    
                    # 分析每列的数据质量
                    print(f"\n列数据分析:")
                    for col in full_df.columns:
                        col_data = full_df[col]
                        print(f"\n列 '{col}':")
                        print(f"  类型: {col_data.dtype}")
                        print(f"  非空值数量: {col_data.notna().sum()}")
                        print(f"  NaN数量: {col_data.isna().sum()}")
                        print(f"  唯一值数量: {col_data.nunique()}")
                        
                        # 如果是数值列，显示统计信息
                        if col_data.dtype in ['int64', 'float64']:
                            print(f"  数值范围: {col_data.min()} - {col_data.max()}")
                            print(f"  平均值: {col_data.mean()}")
                        else:
                            print(f"  前5个唯一值: {col_data.dropna().unique()[:5].tolist()}")
                    
                    # 测试数值转换
                    print(f"\n数值转换测试:")
                    for col in full_df.columns:
                        try:
                            numeric_data = pd.to_numeric(full_df[col], errors='coerce')
                            nan_count = numeric_data.isna().sum()
                            print(f"  '{col}' -> 数值转换后NaN数量: {nan_count}/{len(numeric_data)}")
                            if nan_count < len(numeric_data):
                                valid_data = numeric_data.dropna()
                                print(f"    有效数值范围: {valid_data.min()} - {valid_data.max()}")
                        except Exception as e:
                            print(f"  '{col}' -> 转换失败: {e}")
                    
                    return True
                    
                except UnicodeDecodeError:
                    print(f"✗ 编码 {encoding} 失败")
                    continue
                except Exception as e:
                    print(f"✗ 读取失败: {e}")
                    continue
                    
            print("✗ 所有编码方式都失败")
            return False
        
        # Excel文件的数据分析部分
        if file_ext in ['.xlsx', '.xls']:
            # 分析每列的数据质量
            print(f"\n列数据分析:")
            for col in full_df.columns:
                col_data = full_df[col]
                print(f"\n列 '{col}':")
                print(f"  类型: {col_data.dtype}")
                print(f"  非空值数量: {col_data.notna().sum()}")
                print(f"  NaN数量: {col_data.isna().sum()}")
                print(f"  唯一值数量: {col_data.nunique()}")
                
                # 如果是数值列，显示统计信息
                if col_data.dtype in ['int64', 'float64']:
                    print(f"  数值范围: {col_data.min()} - {col_data.max()}")
                    print(f"  平均值: {col_data.mean()}")
                else:
                    print(f"  前5个唯一值: {col_data.dropna().unique()[:5].tolist()}")
            
            # 测试数值转换
            print(f"\n数值转换测试:")
            for col in full_df.columns:
                try:
                    numeric_data = pd.to_numeric(full_df[col], errors='coerce')
                    nan_count = numeric_data.isna().sum()
                    print(f"  '{col}' -> 数值转换后NaN数量: {nan_count}/{len(numeric_data)}")
                    if nan_count < len(numeric_data):
                        valid_data = numeric_data.dropna()
                        print(f"    有效数值范围: {valid_data.min()} - {valid_data.max()}")
                except Exception as e:
                    print(f"  '{col}' -> 转换失败: {e}")
            
            return True
        
    except Exception as e:
        print(f"✗ 调试过程中出错: {e}")
        return False

def test_problematic_cases():
    """测试各种可能导致问题的数据情况"""
    print("\n=== 测试边界情况 ===")
    
    # 创建各种测试数据
    test_cases = [
        # 包含非数值数据
        pd.DataFrame({
            'Wavelength': [400, 401, 402, 403, 404],
            'Counts': [100, 'abc', 200, None, 300]
        }),
        # 包含特殊字符
        pd.DataFrame({
            'Wavelength(nm)': [400, 401, 402],
            'Counts_值': [100, 200, 300]
        }),
        # 单列数据
        pd.DataFrame({'Data': [1, 2, 3, 4, 5]}),
        # 空数据
        pd.DataFrame({'Col1': [], 'Col2': []})
    ]
    
    for i, df in enumerate(test_cases):
        print(f"\n测试案例 {i+1}:")
        print(f"数据:\n{df}")
        print(f"数据类型:\n{df.dtypes}")
        
        # 测试数值转换
        for col in df.columns:
            try:
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                print(f"'{col}' 转换结果: {numeric_data.values}")
                print(f"NaN位置: {np.isnan(numeric_data.values)}")
            except Exception as e:
                print(f"'{col}' 转换失败: {e}")

if __name__ == "__main__":
    # 测试样本文件
    sample_files = [
        'sample_spectral_data.csv',
        'sample_no_header.csv',
        'sample_spectral_data.xlsx',
        'sample_no_header.xlsx'
    ]
    
    print("开始数据加载调试...")
    
    for file_path in sample_files:
        debug_data_loading(file_path)
        print("-" * 50)
    
    # 测试边界情况
    test_problematic_cases()
    
    print("\n调试完成!")