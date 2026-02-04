import pandas as pd
import os
from pathlib import Path

def split_csv_to_two_column_files(input_file_path, output_folder_name="split_files"):
    """
    将CSV文件按列拆分成多个两列文件
    第一列保持不变，第二列依次是原文件的第2、3、4...列
    
    Args:
        input_file_path (str): 输入CSV文件路径
        output_folder_name (str): 输出文件夹名称
    """
    # 读取CSV文件
    df = pd.read_csv(input_file_path)
    
    # 获取输入文件所在目录
    input_dir = Path(input_file_path).parent
    
    # 创建输出文件夹
    output_dir = input_dir / output_folder_name
    output_dir.mkdir(exist_ok=True)
    
    print(f"输入文件: {input_file_path}")
    print(f"输出目录: {output_dir}")
    print(f"原始数据形状: {df.shape}")
    
    # 第一列作为固定列
    first_column = df.iloc[:, 0]
    first_column_name = df.columns[0]
    
    # 从第二列开始拆分
    for i in range(1, len(df.columns)):
        # 创建新的DataFrame，包含第一列和当前列
        new_df = pd.DataFrame({
            first_column_name: first_column,
            df.columns[i]: df.iloc[:, i]
        })
        
        # 生成文件名
        output_filename = f"split_file_{i}.csv"
        output_path = output_dir / output_filename
        
        # 保存文件
        new_df.to_csv(output_path, index=False, encoding='utf-8')
        
        print(f"已创建: {output_filename} (包含列: {first_column_name}, {df.columns[i]})")
    
    print(f"\n完成！共生成 {len(df.columns)-1} 个文件")
    print(f"文件保存在: {output_dir}")

if __name__ == "__main__":
    # 设置输入文件路径
    input_file = r"C:\Users\lenovo\Desktop\middle2026_01_16_1_corrected.csv"
    
    # 执行拆分
    split_csv_to_two_column_files(input_file)