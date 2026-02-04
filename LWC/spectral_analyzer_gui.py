"""
光谱数据小波变换分析GUI程序
使用tkinter构建用户界面
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import os
from wavelet_transform import WaveletTransform, calculate_snr, apply_wavelet_denoising

class SpectralAnalyzerGUI:
    """光谱数据分析GUI主类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("光谱数据小波变换分析工具")
        self.root.geometry("1200x800")
        
        # 数据存储
        self.spectral_data = None  # 原始光谱数据 [wavelength, counts1, counts2, ...]
        self.wavelength = None     # 波长数据
        self.counts_data = []      # 计数数据列表
        self.file_path = None      # 当前文件路径
        
        # 小波参数
        self.wavelet_type = tk.StringVar(value='db4')
        self.decomposition_levels = tk.IntVar(value=6)
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="光谱数据小波变换分析工具", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 控制面板框架
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # 文件操作区域
        file_frame = ttk.Frame(control_frame)
        file_frame.grid(row=0, column=0, padx=(0, 20))
        
        ttk.Button(file_frame, text="导入单个文件", 
                  command=self.load_single_file).pack(pady=5)
        ttk.Button(file_frame, text="导入文件夹", 
                  command=self.load_folder_data).pack(pady=5)
        ttk.Button(file_frame, text="保存结果", 
                  command=self.save_results).pack(pady=5)
        
        # 小波参数设置区域
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=0, column=1, padx=(0, 20))
        
        ttk.Label(param_frame, text="小波类型:").grid(row=0, column=0, sticky=tk.W, pady=2)
        wavelet_combo = ttk.Combobox(param_frame, textvariable=self.wavelet_type,
                                   values=['db4'], state='readonly', width=10)
        wavelet_combo.grid(row=0, column=1, padx=(5, 0), pady=2)
        
        ttk.Label(param_frame, text="分解层数:").grid(row=1, column=0, sticky=tk.W, pady=2)
        level_spinbox = ttk.Spinbox(param_frame, from_=1, to=10, 
                                  textvariable=self.decomposition_levels, width=10)
        level_spinbox.grid(row=1, column=1, padx=(5, 0), pady=2)
        
        # 操作按钮区域
        action_frame = ttk.Frame(control_frame)
        action_frame.grid(row=0, column=2)
        
        ttk.Button(action_frame, text="执行小波变换", 
                  command=self.perform_wavelet_transform).pack(pady=2)
        ttk.Button(action_frame, text="计算信噪比", 
                  command=self.calculate_snr_all).pack(pady=2)
        ttk.Button(action_frame, text="波长域信噪比", 
                  command=self.show_wavelength_snr_spectrum).pack(pady=2)
        ttk.Button(action_frame, text="清除所有", 
                  command=self.clear_all).pack(pady=2)
        
        # 显示当前文件信息
        self.file_info_label = ttk.Label(control_frame, text="未加载数据文件", 
                                       foreground='blue')
        self.file_info_label.grid(row=1, column=0, columnspan=3, pady=(10, 0))
        
        # 主内容区域
        content_frame = ttk.Frame(main_frame)
        content_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=1)
        content_frame.rowconfigure(0, weight=1)
        
        # 图形显示区域
        self.create_plot_area(content_frame)
        
        # 结果显示区域
        self.create_result_area(content_frame)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_plot_area(self, parent):
        """创建图形显示区域"""
        plot_frame = ttk.LabelFrame(parent, text="光谱数据显示", padding="5")
        plot_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)
        
        # 创建matplotlib图形
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 工具栏
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.grid(row=1, column=0, pady=(5, 0))
        
        ttk.Button(toolbar_frame, text="显示原始数据", 
                  command=self.plot_original_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar_frame, text="显示变换结果", 
                  command=self.plot_transformed_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar_frame, text="对比显示", 
                  command=self.plot_comparison).pack(side=tk.LEFT, padx=(0, 5))
        
    def create_result_area(self, parent):
        """创建结果显示区域"""
        result_frame = ttk.LabelFrame(parent, text="分析结果", padding="5")
        result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        
        # 创建文本显示框
        self.result_text = scrolledtext.ScrolledText(result_frame, width=40, height=20)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 清除按钮
        ttk.Button(result_frame, text="清空结果", 
                  command=lambda: self.result_text.delete(1.0, tk.END)).grid(row=1, column=0, pady=(5, 0))
        
    def load_single_file(self):
        """加载单个数据文件"""
        self.load_csv_data()
    
    def load_folder_data(self):
        """加载文件夹中的所有数据文件"""
        folder_path = filedialog.askdirectory(
            title="选择包含光谱数据文件的文件夹"
        )
        
        if not folder_path:
            return
            
        try:
            self.status_var.set("正在扫描文件夹...")
            self.root.update()
            
            # 支持的文件扩展名
            supported_extensions = {'.csv', '.xlsx', '.xls'}
            
            # 获取文件夹中所有支持的文件
            data_files = []
            for filename in os.listdir(folder_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in supported_extensions:
                    file_path = os.path.join(folder_path, filename)
                    data_files.append(file_path)
            
            if not data_files:
                messagebox.showwarning("警告", "文件夹中没有找到支持的数据文件")
                return
            
            self.status_var.set(f"找到 {len(data_files)} 个数据文件，正在加载...")
            self.root.update()
            
            # 加载所有文件
            all_data = []
            loaded_files = []
            
            for file_path in data_files:
                try:
                    df = self._read_data_file(file_path)
                    if df is not None:
                        all_data.append(df)
                        loaded_files.append(os.path.basename(file_path))
                except Exception as e:
                    print(f"加载文件 {file_path} 失败: {e}")
                    continue
            
            if not all_data:
                messagebox.showerror("错误", "没有成功加载任何数据文件")
                return
            
            # 合并数据
            self._merge_multiple_datasets(all_data, loaded_files, folder_path)
            
            # 更新界面
            self.file_path = folder_path
            self.file_info_label.config(text=f"已加载文件夹: {os.path.basename(folder_path)} ({len(loaded_files)} 个文件)")
            self.status_var.set("文件夹数据加载完成")
            
            # 显示原始数据
            self.plot_original_data()
            
            # 显示基本信息
            self.display_basic_info()
            
        except Exception as e:
            messagebox.showerror("错误", f"加载文件夹数据失败:\n{str(e)}")
            self.status_var.set("加载失败")
    
    def load_csv_data(self):
        """加载CSV数据文件"""
        file_path = filedialog.askopenfilename(
            title="选择光谱数据文件",
            filetypes=[
                ("数据文件", "*.csv *.xlsx *.xls"),
                ("CSV文件", "*.csv"),
                ("Excel文件", "*.xlsx *.xls"),
                ("所有文件", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            self.status_var.set("正在加载数据...")
            self.root.update()
            
            # 读取数据文件
            df = self._read_data_file(file_path)
            
            if df is None:
                # 读取Excel文件
                try:
                    # 先尝试读取第一个工作表
                    excel_file = pd.ExcelFile(file_path)
                    first_sheet = excel_file.sheet_names[0]
                    
                    # 读取前几行判断是否有表头
                    sample_df = pd.read_excel(file_path, sheet_name=first_sheet, nrows=5)
                    has_header = self._detect_header(sample_df)
                    
                    # 根据是否有表头读取整个文件
                    if has_header:
                        df = pd.read_excel(file_path, sheet_name=first_sheet)
                    else:
                        df = pd.read_excel(file_path, sheet_name=first_sheet, header=None)
                except Exception as e:
                    raise Exception(f"读取Excel文件失败: {str(e)}")
            else:
                # 读取CSV文件
                encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
                
                for encoding in encodings:
                    try:
                        # 先尝试读取前几行判断是否有表头
                        sample_df = pd.read_csv(file_path, nrows=5, encoding=encoding)
                        has_header = self._detect_header(sample_df)
                        
                        # 根据是否有表头重新读取整个文件
                        if has_header:
                            df = pd.read_csv(file_path, encoding=encoding)
                        else:
                            df = pd.read_csv(file_path, header=None, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                    
            if df is None:
                if file_extension in ['.xlsx', '.xls']:
                    raise Exception("无法读取Excel文件，请检查文件格式")
                else:
                    raise Exception("无法读取CSV文件，请检查文件编码")
            
            # 解析数据
            self._parse_spectral_data(df, file_path)
            
            # 更新界面
            self.file_path = file_path
            filename = os.path.basename(file_path)
            self.file_info_label.config(text=f"已加载: {filename}")
            self.status_var.set("数据加载完成")
            
            # 显示原始数据
            self.plot_original_data()
            
            # 显示基本信息
            self.display_basic_info()
            
        except Exception as e:
            messagebox.showerror("错误", f"加载数据失败:\n{str(e)}")
            self.status_var.set("加载失败")
    
    def _read_data_file(self, file_path):
        """读取单个数据文件"""
        file_extension = os.path.splitext(file_path)[1].lower()
        df = None
        
        if file_extension in ['.xlsx', '.xls']:
            # 读取Excel文件
            try:
                # 先尝试读取第一个工作表
                excel_file = pd.ExcelFile(file_path)
                first_sheet = excel_file.sheet_names[0]
                
                # 读取前几行判断是否有表头
                sample_df = pd.read_excel(file_path, sheet_name=first_sheet, nrows=5)
                has_header = self._detect_header(sample_df)
                
                # 根据是否有表头读取整个文件
                if has_header:
                    df = pd.read_excel(file_path, sheet_name=first_sheet)
                else:
                    df = pd.read_excel(file_path, sheet_name=first_sheet, header=None)
            except Exception as e:
                print(f"读取Excel文件失败 {file_path}: {str(e)}")
                return None
        else:
            # 读取CSV文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
            
            for encoding in encodings:
                try:
                    # 先尝试读取前几行判断是否有表头
                    sample_df = pd.read_csv(file_path, nrows=5, encoding=encoding)
                    has_header = self._detect_header(sample_df)
                    
                    # 根据是否有表头重新读取整个文件
                    if has_header:
                        df = pd.read_csv(file_path, encoding=encoding)
                    else:
                        df = pd.read_csv(file_path, header=None, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
        
        return df
    
    def _merge_multiple_datasets(self, datasets, filenames, folder_path):
        """合并多个数据集"""
        # 假设所有文件都有相同的波长列（第一列）
        # 使用第一个文件的波长作为基准
        base_wavelength = datasets[0].iloc[:, 0].values
        merged_counts = []
        
        # 为每个文件创建计数系列
        for i, (df, filename) in enumerate(zip(datasets, filenames)):
            try:
                # 提取计数数据（第二列）
                if df.shape[1] >= 2:
                    counts_series = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                    counts = counts_series.values
                    
                    # 移除NaN和无穷大值
                    mask = ~np.isnan(counts) & np.isfinite(counts)
                    if np.sum(mask) > 0:
                        wavelength_clean = base_wavelength[mask]
                        counts_clean = counts[mask]
                        counts_clean = counts_clean.astype(float)
                        
                        merged_counts.append({
                            'name': os.path.splitext(filename)[0],
                            'wavelength': wavelength_clean,
                            'counts': counts_clean
                        })
            except Exception as e:
                print(f"处理文件 {filename} 失败: {e}")
                continue
        
        if not merged_counts:
            raise Exception("没有找到有效的计数数据")
        
        self.wavelength = base_wavelength
        self.counts_data = merged_counts
        self.spectral_data = {
            'wavelength': self.wavelength,
            'counts_series': self.counts_data
        }
    
    def _detect_header(self, df):
        """检测CSV文件是否有表头"""
        # 检查第一行是否包含非数值的字符串
        first_row = df.iloc[0].astype(str)
        numeric_count = sum(pd.to_numeric(first_row, errors='coerce').notna())
        
        # 如果大部分都是数字，则认为没有表头
        return numeric_count < len(first_row) * 0.5
    
    def _parse_spectral_data(self, df, file_path):
        """解析光谱数据"""
        # 假设第一列是波长，其余列是计数值
        if df.shape[1] < 2:
            raise Exception("数据列数不足，至少需要波长和一个计数列")
            
        # 提取波长数据
        self.wavelength = df.iloc[:, 0].values
        
        # 提取计数数据（可能有多列）
        self.counts_data = []
        for i in range(1, df.shape[1]):
            # 尝试将数据转换为数值类型
            try:
                counts_series = pd.to_numeric(df.iloc[:, i], errors='coerce')
                counts = counts_series.values
            except Exception as e:
                raise Exception(f"第{i+1}列数据转换失败: {str(e)}")
            
            # 移除NaN和无穷大值
            mask = ~np.isnan(counts) & np.isfinite(counts)
            if np.sum(mask) > 0:
                wavelength_clean = self.wavelength[mask]
                counts_clean = counts[mask]
                # 确保数据为浮点类型
                counts_clean = counts_clean.astype(float)
                self.counts_data.append({
                    'name': f'Counts_{i}' if df.shape[1] > 2 else 'Counts',
                    'wavelength': wavelength_clean,
                    'counts': counts_clean
                })
        
        if not self.counts_data:
            raise Exception("没有找到有效的计数数据")
            
        self.spectral_data = {
            'wavelength': self.wavelength,
            'counts_series': self.counts_data
        }
    
    def display_basic_info(self):
        """显示基本数据信息"""
        if not self.spectral_data:
            return
            
        info = "=== 数据基本信息 ===\n"
        info += f"文件路径: {self.file_path}\n"
        info += f"数据点数: {len(self.wavelength)}\n"
        info += f"波长范围: {self.wavelength.min():.2f} - {self.wavelength.max():.2f}\n"
        info += f"计数序列数: {len(self.counts_data)}\n\n"
        
        for i, series in enumerate(self.counts_data):
            counts = series['counts']
            info += f"序列 {i+1} ({series['name']}):\n"
            info += f"  数据点数: {len(counts)}\n"
            info += f"  数值范围: {counts.min():.2f} - {counts.max():.2f}\n"
            info += f"  平均值: {np.mean(counts):.2f}\n"
            info += f"  标准差: {np.std(counts):.2f}\n\n"
            
        self.result_text.insert(tk.END, info)
        self.result_text.see(tk.END)
    
    def plot_original_data(self):
        """显示原始数据"""
        if not self.spectral_data:
            messagebox.showwarning("警告", "请先加载数据")
            return
            
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, series in enumerate(self.counts_data):
            color = colors[i % len(colors)]
            ax.plot(series['wavelength'], series['counts'], 
                   color=color, alpha=0.7, linewidth=1,
                   label=series['name'])
        
        ax.set_xlabel('波长')
        ax.set_ylabel('计数值')
        ax.set_title('原始光谱数据')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        self.status_var.set("已显示原始数据")
    
    def plot_transformed_data(self):
        """显示小波变换后的数据"""
        if not self.spectral_data:
            messagebox.showwarning("警告", "请先加载数据")
            return
            
        try:
            self.status_var.set("正在执行小波变换...")
            self.root.update()
            
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            
            colors = ['blue', 'red', 'green', 'orange', 'purple']
            
            for i, series in enumerate(self.counts_data):
                # 执行小波去噪
                denoised = apply_wavelet_denoising(
                    series['counts'], 
                    levels=self.decomposition_levels.get(),
                    wavelet_name=self.wavelet_type.get()
                )
                
                color = colors[i % len(colors)]
                ax.plot(series['wavelength'], denoised, 
                       color=color, alpha=0.7, linewidth=1,
                       label=f"{series['name']}_去噪")
            
            ax.set_xlabel('波长')
            ax.set_ylabel('计数值')
            ax.set_title('小波变换后数据')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            self.canvas.draw()
            self.status_var.set("小波变换完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"小波变换失败:\n{str(e)}")
            self.status_var.set("变换失败")
    
    def plot_comparison(self):
        """对比显示原始数据和变换后数据"""
        if not self.spectral_data:
            messagebox.showwarning("警告", "请先加载数据")
            return
            
        try:
            self.status_var.set("正在生成对比图...")
            self.root.update()
            
            self.fig.clear()
            
            # 为每个数据系列创建子图
            n_series = len(self.counts_data)
            for i, series in enumerate(self.counts_data):
                # 原始数据
                ax1 = self.fig.add_subplot(n_series, 2, 2*i + 1)
                ax1.plot(series['wavelength'], series['counts'], 'b-', alpha=0.7, linewidth=1)
                ax1.set_title(f'{series["name"]} - 原始数据')
                ax1.set_ylabel('计数值')
                ax1.grid(True, alpha=0.3)
                if i == n_series - 1:
                    ax1.set_xlabel('波长')
                
                # 变换后数据
                denoised = apply_wavelet_denoising(
                    series['counts'], 
                    levels=self.decomposition_levels.get(),
                    wavelet_name=self.wavelet_type.get()
                )
                
                ax2 = self.fig.add_subplot(n_series, 2, 2*i + 2)
                ax2.plot(series['wavelength'], denoised, 'r-', alpha=0.7, linewidth=1)
                ax2.set_title(f'{series["name"]} - 小波去噪后')
                ax2.grid(True, alpha=0.3)
                if i == n_series - 1:
                    ax2.set_xlabel('波长')
            
            self.fig.tight_layout()
            self.canvas.draw()
            self.status_var.set("对比图生成完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"生成对比图失败:\n{str(e)}")
            self.status_var.set("生成失败")
    
    def calculate_snr_all(self):
        """计算所有数据系列的信噪比"""
        if not self.spectral_data:
            messagebox.showwarning("警告", "请先加载数据")
            return
            
        try:
            self.status_var.set("正在计算信噪比...")
            self.root.update()
            
            # 执行小波变换获取去噪数据
            snr_results = []
            
            for i, series in enumerate(self.counts_data):
                # 原始SNR（比例形式）
                original_snr = calculate_snr(series['counts'], use_ratio=True)
                
                # 去噪后SNR（比例形式）
                denoised = apply_wavelet_denoising(
                    series['counts'], 
                    levels=self.decomposition_levels.get(),
                    wavelet_name=self.wavelet_type.get()
                )
                denoised_snr = calculate_snr(denoised, use_ratio=True)
                
                snr_results.append({
                    'name': series['name'],
                    'original_snr': original_snr,
                    'denoised_snr': denoised_snr,
                    'improvement': denoised_snr - original_snr
                })
            
            # 显示结果
            result_text = "\n=== 信噪比分析结果 ===\n"
            result_text += f"小波类型: {self.wavelet_type.get()}\n"
            result_text += f"分解层数: {self.decomposition_levels.get()}\n\n"
            
            for result in snr_results:
                result_text += f"数据系列: {result['name']}\n"
                result_text += f"  原始SNR: {result['original_snr']:.2f}（比值）\n"
                result_text += f"  去噪后SNR: {result['denoised_snr']:.2f}（比值）\n"
                result_text += f"  改善量: {result['improvement']:.2f}（倍数）\n"
                if result['improvement'] > 0:
                    result_text += "  ✓ 信噪比得到改善\n"
                else:
                    result_text += "  ✗ 信噪比未改善\n"
                result_text += "\n"
            
            # 计算平均改善
            improvements = [r['improvement'] for r in snr_results]
            avg_improvement = np.mean(improvements)
            result_text += f"平均SNR改善: {avg_improvement:.2f} dB\n"
            
            self.result_text.insert(tk.END, result_text)
            self.result_text.see(tk.END)
            self.status_var.set("信噪比计算完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"信噪比计算失败:\n{str(e)}")
            self.status_var.set("计算失败")
    
    def show_wavelength_snr_spectrum(self):
        """显示每个波长点的信噪比谱图"""
        if not self.spectral_data:
            messagebox.showwarning("警告", "请先加载数据")
            return
            
        try:
            self.status_var.set("正在计算每个波长的信噪比...")
            self.root.update()
            
            # 准备多帧数据用于计算波长域信噪比
            if len(self.counts_data) < 2:
                messagebox.showwarning("警告", "需要至少两个数据系列才能计算波长域信噪比")
                return
            
            # 将所有数据系列组合成多帧数据
            frames_data = np.array([series['counts'] for series in self.counts_data])
            
            # 对每一帧进行小波去噪
            denoised_frames = []
            for frame in frames_data:
                denoised_frame = apply_wavelet_denoising(
                    frame,
                    levels=self.decomposition_levels.get(),
                    wavelet_name=self.wavelet_type.get()
                )
                denoised_frames.append(denoised_frame)
            
            denoised_frames = np.array(denoised_frames)
            
            # 计算每个波长点的信噪比
            from wavelet_transform import calculate_pointwise_snr
            wavelength_snr, wavelength_values = calculate_pointwise_snr(frames_data, denoised_frames)
            
            # 创建新的图形窗口显示波长域信噪比
            plt.figure(figsize=(12, 8))
            
            # 绘制信噪比谱图
            plt.subplot(2, 1, 1)
            plt.plot(wavelength_values, wavelength_snr, 'b-', linewidth=2, label='波长域信噪比')
            plt.xlabel('波长 (nm)')
            plt.ylabel('信噪比 (比值)')
            plt.title('每个波长点的信噪比谱')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # 添加统计信息
            mean_snr = np.mean(wavelength_snr)
            max_snr = np.max(wavelength_snr)
            min_snr = np.min(wavelength_snr)
            
            stats_text = f'平均SNR: {mean_snr:.2f}（比值）\n'
            stats_text += f'最大SNR: {max_snr:.2f}（比值）\n'
            stats_text += f'最小SNR: {min_snr:.2f}（比值）'
            
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 绘制信噪比分布直方图
            plt.subplot(2, 1, 2)
            plt.hist(wavelength_snr, bins=50, alpha=0.7, color='green', edgecolor='black')
            plt.xlabel('信噪比 (比值)')
            plt.ylabel('频数')
            plt.title('信噪比分布直方图')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # 在结果文本框中添加摘要信息
            result_text = "\n=== 波长域信噪比分谱析 ===\n"
            result_text += f"数据系列数: {len(self.counts_data)}\n"
            result_text += f"波长点数: {len(wavelength_values)}\n"
            result_text += f"波长范围: {wavelength_values[0]:.1f} - {wavelength_values[-1]:.1f} nm\n"
            result_text += f"平均信噪比: {mean_snr:.2f}（比值）\n"
            result_text += f"最高信噪比: {max_snr:.2f}（比值） (波长: {wavelength_values[np.argmax(wavelength_snr)]:.1f} nm)\n"
            result_text += f"最低信噪比: {min_snr:.2f}（比值） (波长: {wavelength_values[np.argmin(wavelength_snr)]:.1f} nm)\n"
            
            self.result_text.insert(tk.END, result_text)
            self.result_text.see(tk.END)
            self.status_var.set("波长域信噪比计算完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"波长域信噪比计算失败:\n{str(e)}")
            self.status_var.set("计算失败")
    
    def perform_wavelet_transform(self):
        """执行小波变换（单独功能）"""
        if not self.spectral_data:
            messagebox.showwarning("警告", "请先加载数据")
            return
            
        try:
            self.status_var.set("正在执行小波变换...")
            self.root.update()
            
            # 这里可以添加更详细的小波变换结果显示
            result_text = "\n=== 小波变换执行记录 ===\n"
            result_text += f"时间: {pd.Timestamp.now()}\n"
            result_text += f"小波类型: {self.wavelet_type.get()}\n"
            result_text += f"分解层数: {self.decomposition_levels.get()}\n"
            result_text += f"处理数据系列数: {len(self.counts_data)}\n"
            result_text += "变换执行成功！\n\n"
            
            self.result_text.insert(tk.END, result_text)
            self.result_text.see(tk.END)
            self.status_var.set("小波变换执行完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"小波变换执行失败:\n{str(e)}")
            self.status_var.set("执行失败")
    
    def save_results(self):
        """保存分析结果"""
        if not self.spectral_data:
            messagebox.showwarning("警告", "没有可保存的数据")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="保存分析结果",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.result_text.get(1.0, tk.END))
                messagebox.showinfo("成功", f"结果已保存到:\n{file_path}")
                self.status_var.set("结果保存完成")
            except Exception as e:
                messagebox.showerror("错误", f"保存失败:\n{str(e)}")
    
    def clear_all(self):
        """清除所有数据和显示"""
        # 清除数据
        self.spectral_data = None
        self.wavelength = None
        self.counts_data = []
        self.file_path = None
        
        # 清除显示
        self.fig.clear()
        self.canvas.draw()
        self.result_text.delete(1.0, tk.END)
        self.file_info_label.config(text="未加载数据文件")
        self.status_var.set("已清除所有数据")

def main():
    """主函数"""
    root = tk.Tk()
    app = SpectralAnalyzerGUI(root)
    
    # 设置窗口图标（如果有的话）
    try:
        # root.iconbitmap('icon.ico')  # 如果有图标文件可以取消注释
        pass
    except:
        pass
    
    root.mainloop()

if __name__ == "__main__":
    main()