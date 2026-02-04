# -*- coding: utf-8 -*-
"""
光谱小波变换去噪程序
基于C语言实现的精确复现版本
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt  # 小波变换库
from scipy import signal
import math

# ----------------------数学基础函数（复现C语言逻辑）----------------------
def newton_sqrt(a):
    """
    牛顿迭代法求平方根（复现C语言qSqrt函数）
    仅使用基本运算，符合嵌入式系统要求
    """
    if a < 0:
        return 0.0
    if a == 0:
        return 0.0
    x = a / 2  # 初始猜测值
    for _ in range(5):  # 5次迭代达到足够精度
        x = (x + a / x) / 2
    return x

def calc_std_c_style(data_array):
    """
    复现C语言calcStdWithC函数的标准差计算
    参数：data_array - numpy数组或列表
    返回：标准差值
    """
    n = len(data_array)
    if n < 2:
        return 0.0
    
    # 计算均值
    sum_x = sum(data_array)
    avg = sum_x / n
    
    # 计算方差
    diff_sum = sum((x - avg) ** 2 for x in data_array)
    variance = diff_sum / (n - 1)
    
    # 开平方得到标准差
    return newton_sqrt(variance)

def soft_threshold_c_style(coef, threshold):
    """
    复现C语言doThreshold函数的软阈值处理
    参数：
        coef - 小波系数
        threshold - 阈值
    返回：处理后的系数
    """
    abs_coef = abs(coef)
    if abs_coef <= threshold:
        return 0.0
    elif coef > 0:
        return coef - threshold
    else:
        return coef + threshold

# ----------------------小波变换核心类（封装C风格逻辑）----------------------
class SpectralWaveletDenoiser:
    """
    光谱小波去噪器
    完整复现C语言test.c中的doWaveletTransFormWithC函数逻辑
    """
    
    def __init__(self, wavelet='db4', level=6, extension='sym'):
        """
        初始化参数
        wavelet: 小波基函数 ('db4')
        level: 分解层数 (6)
        extension: 延拓方式 ('sym')
        """
        self.wavelet = wavelet
        self.level = level
        self.extension = extension
        
    def adaptive_threshold_calculation(self, last_level_coefs):
        """
        自适应阈值计算（复现C语言核心算法）
        参数：last_level_coefs - 最后一层细节系数（D6）
        返回：计算得到的阈值
        """
        # 步骤1：将系数分成10组
        coef_len = len(last_level_coefs)
        group_num = 10
        step = coef_len // group_num
        
        if step == 0:
            step = 1
            
        # 步骤2：计算每组标准差
        group_stds = []
        for i in range(0, coef_len, step):
            end_idx = min(i + step, coef_len)
            group = last_level_coefs[i:end_idx]
            
            std_val = calc_std_c_style(group)
            # 处理标准差为0的情况
            if std_val == 0 and len(group_stds) > 0:
                std_val = group_stds[-1]
            elif std_val == 0 and len(group_stds) == 0:
                std_val = 1e-10  # 避免初始0值
                
            group_stds.append(std_val)
        
        # 步骤3：计算统计特征
        if len(group_stds) == 0:
            return 0.0
            
        std_avg = sum(group_stds) / len(group_stds)
        std_of_stds = calc_std_c_style(group_stds)
        
        # 避免除零
        if std_of_stds == 0:
            std_of_stds = 1e-10
            
        # 步骤4：计算自适应阈值
        std_coef = 1.3  # C语言中的系数
        b = 10          # C语言中的指数
        
        # 阈值公式：threshold = (std_coef * std_avg/std_of_stds)^b
        ratio = std_coef * (std_avg / std_of_stds)
        threshold = ratio ** b
        
        # 限制阈值上限
        threshold = min(threshold, 1000.0)
        
        return threshold
    
    def denoise_single_spectrum(self, spectrum):
        """
        对单条光谱进行小波去噪（完整复现C语言逻辑）
        参数：spectrum - 一维numpy数组，表示光谱数据
        返回：(去噪后的光谱, 使用的阈值)
        """
        # 输入验证
        if len(spectrum) < 10:
            return spectrum.copy(), 0.0
        
        original_length = len(spectrum)
        
        # 步骤1：小波分解
        try:
            coeffs = pywt.wavedec(spectrum, self.wavelet, level=self.level, mode=self.extension)
            cA = coeffs[0]  # 近似系数
            cDs = coeffs[1:]  # 细节系数列表 [D1, D2, ..., D6]
        except Exception:
            return spectrum.copy(), 0.0
        
        # 步骤2：提取最后一层细节系数用于阈值计算
        last_level_coefs = cDs[-1]  # D6系数
        
        # 步骤3：自适应阈值计算
        threshold = self.adaptive_threshold_calculation(last_level_coefs)
        
        # 步骤4：对所有细节系数进行软阈值处理
        cDs_denoised = []
        for cD in cDs:
            cD_denoised = np.array([soft_threshold_c_style(coef, threshold) for coef in cD])
            cDs_denoised.append(cD_denoised)
        
        # 步骤5：小波重构
        try:
            coeffs_denoised = [cA] + cDs_denoised
            spectrum_denoised = pywt.waverec(coeffs_denoised, self.wavelet, mode=self.extension)
        except Exception:
            return spectrum.copy(), threshold
        
        # 确保输出长度与输入一致
        if len(spectrum_denoised) > original_length:
            spectrum_denoised = spectrum_denoised[:original_length]
        elif len(spectrum_denoised) < original_length:
            # 如果长度不足，用原数据补足
            padding = original_length - len(spectrum_denoised)
            spectrum_denoised = np.pad(spectrum_denoised, (0, padding), mode='edge')
        
        return spectrum_denoised, threshold
    
    def batch_denoise(self, spectra_matrix):
        """
        批量处理多条光谱
        参数：spectra_matrix - 二维numpy数组，形状(光谱数, 波长点数)
        返回：(去噪后的光谱矩阵, 阈值列表)
        """
        denoised_spectra = []
        thresholds = []
        total_spectra = len(spectra_matrix)
        
        for i, spectrum in enumerate(spectra_matrix):
            denoised_spectrum, threshold = self.denoise_single_spectrum(spectrum)
            denoised_spectra.append(denoised_spectrum)
            thresholds.append(threshold)
        
        return np.array(denoised_spectra), thresholds

# ----------------------数据处理和可视化函数----------------------
def detect_header(df):
    """
    检测CSV文件是否有表头
    根据项目规范：检查第一行是否包含非数值的字符串
    改进的检测逻辑：
    1. 检查是否包含明显的文本标识（如'波长'、'wavelength'等）
    2. 检查数值比例
    3. 检查数据类型多样性
    """
    if df.empty:
        return False
    
    first_row = df.iloc[0]
    
    # 检查是否包含常见的表头关键词
    header_keywords = ['波长', 'wavelength', 'lambda', 'nm', '测量', 'intensity', 'value', 'data']
    first_row_str = ' '.join(str(item).lower() for item in first_row)
    
    for keyword in header_keywords:
        if keyword in first_row_str:
            return True
    
    # 检查数值比例
    numeric_count = sum(pd.to_numeric(first_row, errors='coerce').notna())
    
    # 如果大部分都是数字，则认为没有表头
    if numeric_count >= len(first_row) * 0.8:
        return False
    
    # 检查数据类型多样性
    data_types = set()
    for item in first_row:
        if isinstance(item, (int, float)) or (isinstance(item, str) and item.replace('.', '').isdigit()):
            data_types.add('numeric')
        else:
            data_types.add('text')
    
    # 如果包含文本类型且数值比例不高，则认为有表头
    if 'text' in data_types and numeric_count < len(first_row) * 0.7:
        return True
    
    return False

def load_spectral_data(file_path, wavelength_col=None, spectrum_cols=None):
    """
    载入光谱数据文件（改进版本）
    支持CSV和Excel格式，自动识别表头
    
    参数:
    file_path: 文件路径
    wavelength_col: 波长列名（可选，None表示使用第一列）
    spectrum_cols: 光谱列名列表（可选，None表示使用除第一列外的所有列）
    
    返回:
    wavelength: 波长数据数组
    spectra: 光谱数据矩阵（光谱数 × 波长点数）
    spectrum_names: 光谱列名列表
    """
    # 根据文件扩展名选择读取方法（遵循项目规范）
    if file_path.endswith('.csv'):
        # 先用默认方式读取
        df_default = pd.read_csv(file_path)
        has_header = detect_header(df_default)
        
        if has_header:
            df = df_default
        else:
            df = pd.read_csv(file_path, header=None)
            
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
        has_header = True  # Excel通常都有表头
    else:
        raise ValueError("仅支持CSV和Excel格式文件")
    
    # 数据验证
    if df.empty:
        raise ValueError("文件为空")
    
    if df.shape[1] < 2:
        raise ValueError("数据列数不足，至少需要波长和一个测量值列")
    
    # 确定波长列和光谱列
    if wavelength_col is not None:
        # 用户指定了波长列名
        if wavelength_col not in df.columns:
            raise ValueError(f"未找到指定的波长列 '{wavelength_col}'")
        wavelength_idx = list(df.columns).index(wavelength_col)
    else:
        # 默认使用第一列作为波长
        wavelength_idx = 0
        wavelength_col = df.columns[wavelength_idx]
    
    # 提取波长数据
    wavelength_series = df.iloc[:, wavelength_idx]
    
    # 数据类型转换和清洗
    try:
        wavelength = pd.to_numeric(wavelength_series, errors='coerce').values
        # 移除NaN和无穷大值
        valid_mask = np.isfinite(wavelength)
        if not np.any(valid_mask):
            raise ValueError("波长数据中没有有效的数值")
        wavelength = wavelength[valid_mask]
    except Exception as e:
        raise ValueError(f"波长数据转换失败: {str(e)}")
    
    # 确定光谱列
    if spectrum_cols is not None:
        # 用户指定了光谱列
        missing_cols = [col for col in spectrum_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"未找到指定的光谱列: {missing_cols}")
        spectrum_indices = [list(df.columns).index(col) for col in spectrum_cols]
    else:
        # 使用除波长列外的所有列
        spectrum_indices = [i for i in range(df.shape[1]) if i != wavelength_idx]
        spectrum_cols = [df.columns[i] for i in spectrum_indices]
    
    # 提取光谱数据
    spectra_list = []
    valid_spectrum_names = []
    
    for i, col_idx in enumerate(spectrum_indices):
        spectrum_series = df.iloc[:, col_idx]
        try:
            # 数据类型转换
            spectrum_data = pd.to_numeric(spectrum_series, errors='coerce').values
            # 应用波长数据的有效掩码
            if len(spectrum_data) == len(valid_mask):
                spectrum_data = spectrum_data[valid_mask]
            else:
                # 如果长度不匹配，重新处理该列
                spectrum_data = pd.to_numeric(spectrum_series, errors='coerce').dropna().values
                if len(spectrum_data) != len(wavelength):
                    continue
            
            # 检查是否包含有效数据
            if np.any(np.isfinite(spectrum_data)):
                spectra_list.append(spectrum_data)
                valid_spectrum_names.append(spectrum_cols[i])
        except Exception:
            continue
    
    if not spectra_list:
        raise ValueError("没有找到有效的光谱数据列")
    
    # 转换为numpy数组
    spectra = np.array(spectra_list)
    
    return wavelength, spectra, valid_spectrum_names

def plot_spectrum_comparison(wavelength, spectrum_before, spectrum_after, 
                           title="光谱小波去噪前后对比", save_path=None):
    """
    绘制单条光谱去噪前后对比图
    """
    plt.figure(figsize=(12, 8))
    
    # 主对比图
    plt.subplot(2, 1, 1)
    plt.plot(wavelength, spectrum_before, 'o-', color='#ff7f0e', alpha=0.7, 
             linewidth=1.5, markersize=2, label='去噪前')
    plt.plot(wavelength, spectrum_after, 'o-', color='#1f77b4', linewidth=2, 
             markersize=2, label='去噪后')
    plt.xlabel('波长 (nm)', fontsize=12)
    plt.ylabel('强度值', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 差值图
    plt.subplot(2, 1, 2)
    difference = spectrum_after - spectrum_before
    plt.plot(wavelength, difference, 'o-', color='#2ca02c', linewidth=1.5, 
             markersize=2, label='去噪差值')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('波长 (nm)', fontsize=12)
    plt.ylabel('强度差值', fontsize=12)
    plt.title('去噪前后差值', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()

def calculate_snr_analysis(spectra_original, spectra_denoised, wavelength):
    """
    计算信噪比分析
    """
    # 计算噪声功率（去噪前后的差异）
    noise_power = np.var(spectra_original - spectra_denoised, axis=0)
    signal_power = np.var(spectra_denoised, axis=0)
    
    # 信噪比计算（分母加小量避免除零）
    snr_db = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
    
    return snr_db

def plot_snr_analysis(wavelength, snr_db, save_path=None):
    """
    绘制信噪比分析图
    """
    plt.figure(figsize=(12, 6))
    plt.plot(wavelength, snr_db, 'o-', color='#d62728', linewidth=2, 
             markersize=3, label='信噪比 (dB)')
    
    # 添加统计信息
    mean_snr = np.mean(snr_db)
    plt.axhline(y=mean_snr, color='r', linestyle='--', alpha=0.7, 
                label=f'平均SNR: {mean_snr:.2f} dB')
    
    plt.xlabel('波长 (nm)', fontsize=12)
    plt.ylabel('信噪比 (dB)', fontsize=12)
    plt.title('不同波长位置的信噪比分析', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SNR分析图已保存至: {save_path}")
    
    plt.show()
    
    return mean_snr

# ----------------------主程序入口----------------------
def main(file_path, wavelength_col='wavelength', spectrum_cols=None, 
         wavelet='db4', level=6, save_results=True):
    """
    主函数：完整的光谱小波去噪分析流程
    """
    print("=" * 60)
    print("光谱小波变换去噪分析程序")
    print("=" * 60)
    
    try:
        # 1. 数据载入
        print("\n步骤1: 载入光谱数据...")
        wavelength, spectra_raw, spectrum_names = load_spectral_data(
            file_path, wavelength_col, spectrum_cols
        )
        
        # 2. 初始化去噪器
        print(f"\n步骤2: 初始化小波去噪器...")
        print(f"  - 小波基函数: {wavelet}")
        print(f"  - 分解层数: {level}")
        denoiser = SpectralWaveletDenoiser(wavelet=wavelet, level=level)
        
        # 3. 执行去噪处理
        print(f"\n步骤3: 执行小波去噪处理...")
        spectra_denoised = denoiser.batch_denoise(spectra_raw)
        
        # 4. 结果可视化（显示第一条光谱的对比）
        print(f"\n步骤4: 生成可视化结果...")
        if len(spectra_raw) > 0:
            plot_spectrum_comparison(
                wavelength, 
                spectra_raw[0], 
                spectra_denoised[0],
                title=f"第1条光谱去噪效果对比 ({spectrum_names[0]})"
            )
        
        # 5. 信噪比分析（如果有多条光谱）
        if len(spectra_raw) >= 2:
            print(f"\n步骤5: 进行信噪比分析...")
            snr_db = calculate_snr_analysis(spectra_raw, spectra_denoised, wavelength)
            mean_snr = plot_snr_analysis(wavelength, snr_db)
            print(f"平均信噪比: {mean_snr:.2f} dB")
        
        # 6. 保存结果
        if save_results:
            print(f"\n步骤6: 保存处理结果...")
            # 创建结果DataFrame
            result_df = pd.DataFrame({'wavelength': wavelength})
            
            for i, name in enumerate(spectrum_names):
                result_df[f'{name}_original'] = spectra_raw[i]
                result_df[f'{name}_denoised'] = spectra_denoised[i]
            
            # 保存文件
            output_path = file_path.replace('.csv', '_denoised.csv').replace('.xlsx', '_denoised.xlsx')
            if output_path.endswith('.csv'):
                result_df.to_csv(output_path, index=False)
            else:
                result_df.to_excel(output_path, index=False)
            
            print(f"去噪结果已保存至: {output_path}")
        
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)
        
        return {
            'wavelength': wavelength,
            'original_spectra': spectra_raw,
            'denoised_spectra': spectra_denoised,
            'spectrum_names': spectrum_names
        }
        
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        raise

# ----------------------运行示例----------------------
if __name__ == "__main__":
    # 配置参数
    FILE_PATH = "your_spectral_data.csv"  # 替换为你的光谱数据文件路径
    WAVELENGTH_COL = "wavelength"         # 波长列名
    SPECTRUM_COLS = None                  # 光谱列名（None表示自动识别）
    
    # 运行主程序
    try:
        results = main(
            file_path=FILE_PATH,
            wavelength_col=WAVELENGTH_COL,
            spectrum_cols=SPECTRUM_COLS,
            wavelet='db4',
            level=6,
            save_results=True
        )
    except Exception as e:
        print(f"程序运行失败: {e}")
        print("请检查数据文件路径和格式是否正确")
