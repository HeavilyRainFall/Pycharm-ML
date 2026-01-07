import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端以支持GUI显示
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from datetime import datetime
import base64
from io import BytesIO
import webbrowser
from pathlib import Path
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # 导入FigureCanvasTkAgg


def load_csv_data(filepath):
    """
    加载CSV文件，自动检测是否有表头
    """
    # 尝试读取前几行来判断是否有表头
    df = pd.read_csv(filepath, header=None)
    
    # 检查第一行是否是数字（如果是，则没有表头）
    try:
        float(df.iloc[0, 0])  # 如果第一列第一个值可以转换为浮点数，则无表头
        df.columns = ['wavelength', 'value']  # 手动指定列名
    except ValueError:
        # 第一列不能转换为浮点数，说明有表头
        df.columns = ['wavelength', 'value']
        
    return df


def calculate_snr_and_dynamic_range(snr_folder_path):
    """
    计算信噪比和动态范围
    """
    d_folder = os.path.join(snr_folder_path, 'D')
    s_folder = os.path.join(snr_folder_path, 'S')
    
    # 检查文件夹是否存在
    if not os.path.exists(d_folder) or not os.path.exists(s_folder):
        raise FileNotFoundError("D或S文件夹不存在")
    
    # 获取所有D和S文件
    d_files = sorted([f for f in os.listdir(d_folder) if f.endswith('.csv')])
    s_files = sorted([f for f in os.listdir(s_folder) if f.endswith('.csv')])
    
    print(f"D文件夹中的文件数量: {len(d_files)}")
    print(f"S文件夹中的文件数量: {len(s_files)}")
    
    # 检查文件数量是否一致
    if len(d_files) != len(s_files):
        raise ValueError("D和S文件夹中的文件数量不一致！")
    
    # 加载第一个文件以获取波长信息
    sample_df_d = load_csv_data(os.path.join(d_folder, d_files[0]))
    wavelengths = sample_df_d['wavelength'].values
    
    # 验证所有文件的波长是否一致
    for file in d_files[1:]:
        df = load_csv_data(os.path.join(d_folder, file))
        if not np.allclose(wavelengths, df['wavelength'].values, rtol=1e-5):
            raise ValueError(f"文件 {file} 的波长与参考波长不匹配！")
    
    for file in s_files:
        df = load_csv_data(os.path.join(s_folder, file))
        if not np.allclose(wavelengths, df['wavelength'].values, rtol=1e-5):
            raise ValueError(f"文件 {file} 的波长与参考波长不匹配！")
    
    # 准备存储数据的数组
    d_data = np.zeros((len(d_files), len(wavelengths)))
    s_data = np.zeros((len(s_files), len(wavelengths)))
    
    # 加载D数据
    for i, file in enumerate(d_files):
        df = load_csv_data(os.path.join(d_folder, file))
        d_data[i, :] = df['value'].values
    
    # 加载S数据
    for i, file in enumerate(s_files):
        df = load_csv_data(os.path.join(s_folder, file))
        s_data[i, :] = df['value'].values
    
    # 计算S和D的平均值
    s_mean = np.mean(s_data, axis=0)
    d_mean = np.mean(d_data, axis=0)
    
    # 计算S的样本标准差
    s_std = np.std(s_data, axis=0, ddof=1)  # ddof=1 表示样本标准差
    
    # 计算信噪比 SNR = (S-D)/std
    # 避免除零错误
    snr = np.divide(s_mean - d_mean, s_std, out=np.zeros_like(s_mean), where=s_std!=0)
    
    # 计算动态范围 - 使用D文件夹中连续9帧数据
    dr_list = []
    num_frames_for_dr = 9
    num_valid_windows = len(d_files) - num_frames_for_dr + 1
    
    for i in range(num_valid_windows):
        # 取连续9帧数据
        window_data = d_data[i:i+num_frames_for_dr, :]
        # 计算该窗口内每一点的均值和标准差
        window_mean = np.mean(window_data, axis=0)
        window_std = np.std(window_data, axis=0, ddof=1)
        # 计算动态范围 DR = (65535 - D) / std
        dr = np.divide(65535 - window_mean, window_std, out=np.zeros_like(window_mean), where=window_std!=0)
        dr_list.append(dr)
    
    # 对动态范围取平均
    if dr_list:
        dynamic_range = np.mean(dr_list, axis=0)
    else:
        dynamic_range = np.zeros_like(wavelengths)
    
    # 计算统计信息
    snr_stats = {
        'max': np.max(snr),
        'min': np.min(snr),
        'mean': np.mean(snr),
        'median': np.median(snr)
    }
    
    dr_stats = {
        'max': np.max(dynamic_range),
        'min': np.min(dynamic_range),
        'mean': np.mean(dynamic_range),
        'median': np.median(dynamic_range)
    }
    
    return wavelengths, snr, dynamic_range, snr_stats, dr_stats


def find_fwhm(x, y, peak_idx):
    """
    计算峰值的半高全宽(FWHM)
    """
    peak_x = x[peak_idx]
    peak_y = y[peak_idx]
    
    # 半高值
    half_max = peak_y / 2.0
    
    # 找到峰值左侧和右侧第一个低于半高值的点
    left_idx = peak_idx
    right_idx = peak_idx
    
    # 向左搜索
    while left_idx > 0 and y[left_idx] > half_max:
        left_idx -= 1
    
    # 向右搜索
    while right_idx < len(y) - 1 and y[right_idx] > half_max:
        right_idx += 1
    
    # 如果找不到合适的点，返回None
    if left_idx <= 0 or right_idx >= len(y) - 1:
        return None, None, None
    
    # 使用插值找到精确的半高点
    # 左侧半高点
    if y[left_idx + 1] != y[left_idx]:
        left_x_interp = x[left_idx] + (half_max - y[left_idx]) * (x[left_idx + 1] - x[left_idx]) / (y[left_idx + 1] - y[left_idx])
    else:
        left_x_interp = x[left_idx]
    
    # 右侧半高点
    if y[right_idx] != y[right_idx - 1]:
        right_x_interp = x[right_idx - 1] + (half_max - y[right_idx - 1]) * (x[right_idx] - x[right_idx - 1]) / (y[right_idx] - y[right_idx - 1])
    else:
        right_x_interp = x[right_idx]
    
    fwhm = right_x_interp - left_x_interp
    
    return left_x_interp, right_x_interp, fwhm


def analyze_resolution(resolution_folder_path):
    """
    分析分辨率文件，进行寻峰和半高宽计算
    """
    resolution_files = sorted([f for f in os.listdir(resolution_folder_path) if f.endswith('.csv')])
    
    results = []
    
    for file in resolution_files:
        filepath = os.path.join(resolution_folder_path, file)
        df = load_csv_data(filepath)
        
        wavelengths = df['wavelength'].values
        values = df['value'].values
        
        # 寻峰
        peaks, properties = find_peaks(values, height=np.max(values)*0.3, distance=int(len(values)*0.01))
        
        # 如果没找到足够的峰，降低阈值
        if len(peaks) == 0:
            peaks, properties = find_peaks(values, height=np.max(values)*0.1, distance=max(1, int(len(values)*0.02)))
        
        peak_data = []
        for i, peak_idx in enumerate(peaks):
            left_x, right_x, fwhm = find_fwhm(wavelengths, values, peak_idx)
            
            if fwhm is not None and fwhm > 0:
                peak_data.append({
                    'peak_wavelength': wavelengths[peak_idx],
                    'peak_value': values[peak_idx],
                    'fwhm': fwhm,
                    'left_half_max': left_x,
                    'right_half_max': right_x
                })
        
        # 计算当前文件的统计信息
        fwhm_values = [peak['fwhm'] for peak in peak_data if peak['fwhm'] is not None]
        if fwhm_values:
            fwhm_stats = {
                'max': np.max(fwhm_values),
                'min': np.min(fwhm_values),
                'mean': np.mean(fwhm_values),
                'median': np.median(fwhm_values)
            }
        else:
            fwhm_stats = {
                'max': 0,
                'min': 0,
                'mean': 0,
                'median': 0
            }
        
        results.append({
            'filename': file,
            'wavelengths': wavelengths,
            'values': values,
            'peaks': peak_data,
            'fwhm_stats': fwhm_stats
        })
    
    return results


def plot_snr_and_dynamic_range_to_html(wavelengths, snr, dynamic_range):
    """
    将SNR和动态范围绘制成图片并返回base64编码
    """
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 绘制信噪比
    ax1.plot(wavelengths, snr, label='SNR', color='blue')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('SNR')
    ax1.set_title('Signal-to-Noise Ratio vs Wavelength')
    ax1.grid(True)
    ax1.legend()
    
    # 绘制动态范围
    ax2.plot(wavelengths, dynamic_range, label='Dynamic Range', color='red')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Dynamic Range')
    ax2.set_title('Dynamic Range vs Wavelength')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图像到内存
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close(fig)  # 关闭图形以释放内存
    
    return img_base64


def plot_resolution_analysis_to_html(results):
    """
    将分辨率分析绘制成图片并返回base64编码列表
    """
    html_parts = []
    
    for idx, result in enumerate(results):
        wavelengths = result['wavelengths']
        values = result['values']
        peaks = result['peaks']
        
        # 创建当前文件的图形
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(wavelengths, values, label=f"{result['filename']}", linewidth=1)
        
        # 标记峰值和半高宽
        for i, peak in enumerate(peaks):
            # 标记峰值
            ax.plot(peak['peak_wavelength'], peak['peak_value'], 'ro', markersize=8)
            ax.text(peak['peak_wavelength'], peak['peak_value'], 
                    f"Peak: {peak['peak_wavelength']:.2f}\nFWHM: {peak['fwhm']:.2f}",
                    verticalalignment='bottom', horizontalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=9)
            
            # 标记半高宽
            if peak['left_half_max'] is not None and peak['right_half_max'] is not None:
                half_max_val = peak['peak_value'] / 2
                ax.hlines(half_max_val, peak['left_half_max'], peak['right_half_max'], 
                          colors='r', linestyles='--', linewidth=1)
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Value')
        ax.set_title(f'Resolution Analysis - File {idx + 1}: {result["filename"]}')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        # 保存图像到内存
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close(fig)  # 关闭图形以释放内存
        
        html_parts.append(img_base64)
    
    return html_parts


def generate_html_report(snr_stats, dr_stats, resolution_results, snr_plot_img, resolution_plots_imgs, filepath):
    """
    生成HTML报告
    """
    html_content = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>光谱数据分析报告</title>
    <style>
        body {{
            font-family: "Microsoft YaHei", Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #444;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .stats-table th, .stats-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
        }}
        .stats-table th {{
            background-color: #4CAF50;
            color: white;
        }}
        .stats-table tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        .section {{
            margin: 30px 0;
        }}
        .file-section {{
            margin: 20px 0;
            padding: 15px;
            border: 1px solid #eee;
            border-radius: 5px;
            background-color: #fafafa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>光谱数据分析报告</h1>
        <p><strong>生成时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="section">
            <h2>信噪比 (SNR) 统计信息</h2>
            <table class="stats-table">
                <tr>
                    <th>最大值</th>
                    <th>最小值</th>
                    <th>平均值</th>
                    <th>中位数</th>
                </tr>
                <tr>
                    <td>{snr_stats['max']:.4f}</td>
                    <td>{snr_stats['min']:.4f}</td>
                    <td>{snr_stats['mean']:.4f}</td>
                    <td>{snr_stats['median']:.4f}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>动态范围 (Dynamic Range) 统计信息</h2>
            <table class="stats-table">
                <tr>
                    <th>最大值</th>
                    <th>最小值</th>
                    <th>平均值</th>
                    <th>中位数</th>
                </tr>
                <tr>
                    <td>{dr_stats['max']:.4f}</td>
                    <td>{dr_stats['min']:.4f}</td>
                    <td>{dr_stats['mean']:.4f}</td>
                    <td>{dr_stats['median']:.4f}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>SNR 和 动态范围 图表</h2>
            <div class="image-container">
                <img src="data:image/png;base64,{snr_plot_img}" alt="SNR和动态范围图表">
            </div>
        </div>
        
        <div class="section">
            <h2>分辨率分析</h2>
            <p><strong>总文件数:</strong> {len(resolution_results)}</p>
            
            {"".join([
                f'''
                <div class="file-section">
                    <h3>{result["filename"]}</h3>
                    <p><strong>峰值数量:</strong> {len(result["peaks"])}</p>
                    <table class="stats-table">
                        <tr>
                            <th>最大值</th>
                            <th>最小值</th>
                            <th>平均值</th>
                            <th>中位数</th>
                        </tr>
                        <tr>
                            <td>{result["fwhm_stats"]["max"]:.4f}</td>
                            <td>{result["fwhm_stats"]["min"]:.4f}</td>
                            <td>{result["fwhm_stats"]["mean"]:.4f}</td>
                            <td>{result["fwhm_stats"]["median"]:.4f}</td>
                        </tr>
                    </table>
                </div>
                ''' for result in resolution_results
            ])}
        </div>
        
        <div class="section">
            <h2>分辨率分析 图表</h2>
            {"".join([
                f'''
                <div class="file-section">
                    <h3>{result["filename"]}</h3>
                    <div class="image-container">
                        <img src="data:image/png;base64,{resolution_plots_imgs[i]}" alt="{result["filename"]} 分析图表">
                    </div>
                </div>
                ''' for i, result in enumerate(resolution_results)
            ])}
        </div>
    </div>
</body>
</html>'''
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)


class SpectraAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("光谱数据分析程序")
        self.root.geometry("1200x800")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 数据存储
        self.snr_data = None
        self.dr_data = None
        self.wavelengths = None
        self.snr_stats = None
        self.dr_stats = None
        self.resolution_results = None
        
        # 默认路径
        self.default_snr_path = os.path.join(os.path.dirname(__file__), "SNR")
        self.default_res_path = os.path.join(os.path.dirname(__file__), "分辨率")
        
        # 创建GUI组件
        self.create_widgets()
    
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # SNR和动态范围分析按钮
        snr_btn = ttk.Button(control_frame, text="分析SNR和动态范围", command=self.analyze_snr_dr)
        snr_btn.pack(side=tk.LEFT, padx=5)
        
        # 分辨率分析按钮
        res_btn = ttk.Button(control_frame, text="分析分辨率", command=self.analyze_resolution)
        res_btn.pack(side=tk.LEFT, padx=5)
        
        # 生成报告按钮
        report_btn = ttk.Button(control_frame, text="生成报告", command=self.generate_and_save_report)
        report_btn.pack(side=tk.LEFT, padx=5)
        
        # 显示统计信息的标签
        self.stats_label = ttk.Label(control_frame, text="请先进行分析")
        self.stats_label.pack(side=tk.RIGHT, padx=5)
        
        # 笔记本控件（用于分页显示）
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # SNR和动态范围图页面
        snr_frame = ttk.Frame(notebook)
        notebook.add(snr_frame, text="SNR和动态范围")
        
        # 创建matplotlib图形和canvas
        self.snr_fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.snr_canvas = FigureCanvasTkAgg(self.snr_fig, snr_frame)
        self.snr_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 分辨率分析页面
        res_frame = ttk.Frame(notebook)
        notebook.add(res_frame, text="分辨率分析")
        
        # 分辨率分析图形
        self.res_fig, self.res_ax = plt.subplots(figsize=(10, 6))
        self.res_canvas = FigureCanvasTkAgg(self.res_fig, res_frame)
        self.res_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 分辨率分析控制
        res_control_frame = ttk.Frame(res_frame)
        res_control_frame.pack(fill=tk.X)
        
        self.res_file_var = tk.StringVar()
        self.res_file_combo = ttk.Combobox(res_control_frame, textvariable=self.res_file_var, state="readonly")
        self.res_file_combo.pack(side=tk.LEFT, padx=5, pady=5)
        self.res_file_combo.bind('<<ComboboxSelected>>', self.on_res_file_selected)
        
        self.next_btn = ttk.Button(res_control_frame, text="下一页", command=self.next_resolution_file)
        self.next_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.prev_btn = ttk.Button(res_control_frame, text="上一页", command=self.prev_resolution_file)
        self.prev_btn.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def analyze_snr_dr(self):
        # 首先尝试默认路径
        snr_folder_path = self.default_snr_path
        if not os.path.exists(snr_folder_path):
            # 如果默认路径不存在，让用户选择
            snr_folder_path = filedialog.askdirectory(title="选择SNR数据文件夹（包含D和S子文件夹）")
            if not snr_folder_path:
                return
        
        try:
            self.wavelengths, self.snr_data, self.dr_data, self.snr_stats, self.dr_stats = calculate_snr_and_dynamic_range(snr_folder_path)
            
            # 更新图形
            self.ax1.clear()
            self.ax1.plot(self.wavelengths, self.snr_data, label='SNR', color='blue')
            self.ax1.set_xlabel('Wavelength (nm)')
            self.ax1.set_ylabel('SNR')
            self.ax1.set_title('Signal-to-Noise Ratio vs Wavelength')
            self.ax1.grid(True)
            self.ax1.legend()
            
            self.ax2.clear()
            self.ax2.plot(self.wavelengths, self.dr_data, label='Dynamic Range', color='red')
            self.ax2.set_xlabel('Wavelength (nm)')
            self.ax2.set_ylabel('Dynamic Range')
            self.ax2.set_title('Dynamic Range vs Wavelength')
            self.ax2.grid(True)
            self.ax2.legend()
            
            self.snr_canvas.draw()
            
            # 更新统计信息标签
            stats_text = f"SNR - Max: {self.snr_stats['max']:.2f}, Min: {self.snr_stats['min']:.2f}, Mean: {self.snr_stats['mean']:.2f}, Median: {self.snr_stats['median']:.2f} | "
            stats_text += f"DR - Max: {self.dr_stats['max']:.2f}, Min: {self.dr_stats['min']:.2f}, Mean: {self.dr_stats['mean']:.2f}, Median: {self.dr_stats['median']:.2f}"
            self.stats_label.config(text=stats_text)
            
            messagebox.showinfo("完成", "SNR和动态范围分析完成！")
        except Exception as e:
            messagebox.showerror("错误", f"分析过程中出现错误: {str(e)}")
    
    def analyze_resolution(self):
        # 首先尝试默认路径
        resolution_folder_path = self.default_res_path
        if not os.path.exists(resolution_folder_path):
            # 如果默认路径不存在，让用户选择
            resolution_folder_path = filedialog.askdirectory(title="选择分辨率数据文件夹")
            if not resolution_folder_path:
                return
        
        try:
            self.resolution_results = analyze_resolution(resolution_folder_path)
            
            # 更新下拉框
            file_names = [res['filename'] for res in self.resolution_results]
            self.res_file_combo['values'] = file_names
            if file_names:
                self.res_file_combo.current(0)
                self.current_res_index = 0
                self.display_current_resolution_file()
            
            messagebox.showinfo("完成", "分辨率分析完成！")
        except Exception as e:
            messagebox.showerror("错误", f"分析过程中出现错误: {str(e)}")
    
    def display_current_resolution_file(self):
        if not self.resolution_results or self.current_res_index >= len(self.resolution_results):
            return
            
        result = self.resolution_results[self.current_res_index]
        wavelengths = result['wavelengths']
        values = result['values']
        peaks = result['peaks']
        
        self.res_ax.clear()
        self.res_ax.plot(wavelengths, values, label=f"{result['filename']}", linewidth=1)
        
        # 标记峰值和半高宽
        for i, peak in enumerate(peaks):
            # 标记峰值
            self.res_ax.plot(peak['peak_wavelength'], peak['peak_value'], 'ro', markersize=8)
            self.res_ax.text(peak['peak_wavelength'], peak['peak_value'], 
                    f"Peak: {peak['peak_wavelength']:.2f}\nFWHM: {peak['fwhm']:.2f}",
                    verticalalignment='bottom', horizontalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=9)
            
            # 标记半高宽
            if peak['left_half_max'] is not None and peak['right_half_max'] is not None:
                half_max_val = peak['peak_value'] / 2
                self.res_ax.hlines(half_max_val, peak['left_half_max'], peak['right_half_max'], 
                          colors='r', linestyles='--', linewidth=1)
        
        self.res_ax.set_xlabel('Wavelength (nm)')
        self.res_ax.set_ylabel('Value')
        self.res_ax.set_title(f'Resolution Analysis - File {self.current_res_index + 1}/{len(self.resolution_results)}: {result["filename"]}')
        self.res_ax.legend()
        self.res_ax.grid(True)
        
        self.res_canvas.draw()
        
        # 更新统计信息
        fwhm_stats = result['fwhm_stats']
        stats_text = f"当前文件: {result['filename']} | "
        stats_text += f"FWHM - Max: {fwhm_stats['max']:.2f}, Min: {fwhm_stats['min']:.2f}, Mean: {fwhm_stats['mean']:.2f}, Median: {fwhm_stats['median']:.2f}"
        self.stats_label.config(text=stats_text)
    
    def on_res_file_selected(self, event):
        selected_file = self.res_file_var.get()
        for i, res in enumerate(self.resolution_results):
            if res['filename'] == selected_file:
                self.current_res_index = i
                break
        self.display_current_resolution_file()
    
    def next_resolution_file(self):
        if self.resolution_results:
            self.current_res_index = (self.current_res_index + 1) % len(self.resolution_results)
            self.res_file_combo.current(self.current_res_index)
            self.display_current_resolution_file()
    
    def prev_resolution_file(self):
        if self.resolution_results:
            self.current_res_index = (self.current_res_index - 1) % len(self.resolution_results)
            self.res_file_combo.current(self.current_res_index)
            self.display_current_resolution_file()
    
    def generate_and_save_report(self):
        if not self.snr_stats or not self.resolution_results:
            messagebox.showwarning("警告", "请先完成SNR/DR和分辨率分析！")
            return
            
        # 生成图表的base64编码
        snr_plot_img = plot_snr_and_dynamic_range_to_html(
            self.wavelengths, self.snr_data, self.dr_data
        )
        
        resolution_plots_imgs = plot_resolution_analysis_to_html(
            self.resolution_results
        )
        
        # 保存HTML报告
        save_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
            title="保存报告为"
        )
        
        if save_path:
            try:
                generate_html_report(
                    self.snr_stats, 
                    self.dr_stats, 
                    self.resolution_results, 
                    snr_plot_img, 
                    resolution_plots_imgs, 
                    save_path
                )
                messagebox.showinfo("完成", f"报告已保存到: {save_path}")
                
                # 询问用户是否打开报告
                if messagebox.askyesno("打开报告", "是否立即打开生成的报告？"):
                    webbrowser.open(f"file://{os.path.abspath(save_path)}")
            except Exception as e:
                messagebox.showerror("错误", f"生成报告时出现错误: {str(e)}")


def main():
    """
    主函数
    """
    root = tk.Tk()
    app = SpectraAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()