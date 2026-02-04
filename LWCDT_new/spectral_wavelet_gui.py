"""
光谱小波变换GUI程序
基于tkinter的图形界面，实现完整的光谱去噪分析功能
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import os
from datetime import datetime

# 导入我们的核心算法
from spectral_wavelet_denoise import (
    SpectralWaveletDenoiser, 
    load_spectral_data
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SpectralWaveletGUI:
    """光谱小波变换GUI主类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("光谱小波变换去噪分析工具")
        self.root.geometry("1400x900")
        
        # 数据存储
        self.original_data = None      # 原始数据字典 {wavelength, spectra, filenames}
        self.denoised_data = None      # 去噪后数据
        self.batch_results = {}        # 批量处理结果
        self.current_file_path = None  # 当前文件路径
        
        # 处理参数
        self.wavelet_type = tk.StringVar(value='db4')
        self.decomposition_level = tk.IntVar(value=6)
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        """创建GUI组件"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="光谱小波变换去噪分析工具", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # 创建主区域（左右分栏）
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=1)
        
        # 右侧显示区域
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=2)
        
        # 创建各区域
        self.create_control_panel(left_frame)
        self.create_display_area(right_frame)
        self.create_status_bar(main_frame)
        
    def create_control_panel(self, parent):
        """创建控制面板"""
        # 控制面板框架
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.pack(fill=tk.BOTH, expand=True)
        
        # 文件操作区域
        file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="载入单个文件", 
                  command=self.load_single_file, width=20).pack(pady=2)
        ttk.Button(file_frame, text="批量处理文件夹", 
                  command=self.batch_process_folder, width=20).pack(pady=2)
        ttk.Button(file_frame, text="合并处理结果", 
                  command=self.merge_processed_files, width=20).pack(pady=2)
        ttk.Button(file_frame, text="保存结果", 
                  command=self.save_results, width=20).pack(pady=2)
        
        # 参数设置区域
        param_frame = ttk.LabelFrame(control_frame, text="小波参数", padding="5")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 小波类型选择
        wavelet_frame = ttk.Frame(param_frame)
        wavelet_frame.pack(fill=tk.X, pady=2)
        ttk.Label(wavelet_frame, text="小波基函数:").pack(side=tk.LEFT)
        wavelet_combo = ttk.Combobox(wavelet_frame, textvariable=self.wavelet_type,
                                   values=['db4', 'db2', 'db3', 'db5', 'haar'], 
                                   state='readonly', width=12)
        wavelet_combo.pack(side=tk.RIGHT)
        
        # 分解层数设置
        level_frame = ttk.Frame(param_frame)
        level_frame.pack(fill=tk.X, pady=2)
        ttk.Label(level_frame, text="分解层数:").pack(side=tk.LEFT)
        level_spinbox = ttk.Spinbox(level_frame, from_=1, to=10, 
                                  textvariable=self.decomposition_level, 
                                  width=12)
        level_spinbox.pack(side=tk.RIGHT)
        
        # 处理操作区域
        process_frame = ttk.LabelFrame(control_frame, text="处理操作", padding="5")
        process_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(process_frame, text="执行小波去噪", 
                  command=self.perform_wavelet_denoise, width=20).pack(pady=2)
        ttk.Button(process_frame, text="计算信噪比", 
                  command=self.calculate_snr_analysis, width=20).pack(pady=2)
        ttk.Button(process_frame, text="显示对比图", 
                  command=self.show_comparison_plots, width=20).pack(pady=2)
        
        # 显示信息区域
        info_frame = ttk.LabelFrame(control_frame, text="文件信息", padding="5")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_text = scrolledtext.ScrolledText(info_frame, height=8, width=30)
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # 清除按钮
        ttk.Button(control_frame, text="清除所有", 
                  command=self.clear_all, width=20).pack(pady=5)
        
    def create_display_area(self, parent):
        """创建显示区域"""
        # 创建选项卡
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 光谱对比图选项卡
        spectrum_frame = ttk.Frame(notebook)
        notebook.add(spectrum_frame, text="光谱对比")
        self.create_spectrum_plot(spectrum_frame)
        
        # 信噪比分析图选项卡
        snr_frame = ttk.Frame(notebook)
        notebook.add(snr_frame, text="信噪比分析")
        self.create_snr_plot(snr_frame)
        
        # 结果数据选项卡
        result_frame = ttk.Frame(notebook)
        notebook.add(result_frame, text="处理结果")
        self.create_result_display(result_frame)
        
    def create_spectrum_plot(self, parent):
        """创建光谱对比图显示区域"""
        # 图形框架
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建matplotlib图形
        self.spectrum_fig = Figure(figsize=(10, 6), dpi=100)
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_fig, master=plot_frame)
        self.spectrum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 工具栏按钮
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(toolbar_frame, text="显示原始光谱", 
                  command=self.plot_original_spectrum).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar_frame, text="显示去噪光谱", 
                  command=self.plot_denoised_spectrum).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar_frame, text="对比显示", 
                  command=self.plot_spectrum_comparison).pack(side=tk.LEFT, padx=(0, 5))
        
    def create_snr_plot(self, parent):
        """创建信噪比分析图显示区域"""
        # 图形框架
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建matplotlib图形
        self.snr_fig = Figure(figsize=(10, 6), dpi=100)
        self.snr_canvas = FigureCanvasTkAgg(self.snr_fig, master=plot_frame)
        self.snr_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 工具栏按钮
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(toolbar_frame, text="显示SNR谱", 
                  command=self.plot_snr_spectrum).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar_frame, text="批量SNR对比", 
                  command=self.plot_batch_snr_comparison).pack(side=tk.LEFT, padx=(0, 5))
        
    def create_result_display(self, parent):
        """创建结果数据显示区域"""
        # 文本显示框架
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建文本显示框
        self.result_text = scrolledtext.ScrolledText(text_frame, width=80, height=25)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 工具栏按钮
        toolbar_frame = ttk.Frame(text_frame)
        toolbar_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(toolbar_frame, text="清空结果", 
                  command=lambda: self.result_text.delete(1.0, tk.END)).pack(side=tk.LEFT)
        ttk.Button(toolbar_frame, text="导出结果", 
                  command=self.export_results_to_file).pack(side=tk.LEFT, padx=(5, 0))
        
    def create_status_bar(self, parent):
        """创建状态栏"""
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(parent, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(5, 0))
        
    # ==================== 文件操作功能 ====================
    
    def load_single_file(self):
        """载入单个光谱文件"""
        file_path = filedialog.askopenfilename(
            title="选择光谱数据文件",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
            
        try:
            self.status_var.set("正在载入文件...")
            self.root.update()
            
            # 载入数据（使用改进的自动识别功能）
            wavelength, spectra, spectrum_names = load_spectral_data(
                file_path, None, None  # 让函数自动识别波长列和光谱列
            )
            
            # 存储数据
            self.original_data = {
                'wavelength': wavelength,
                'spectra': spectra,
                'filenames': [os.path.basename(file_path)],
                'spectrum_names': spectrum_names
            }
            self.current_file_path = file_path
            
            # 更新界面
            self.update_file_info()
            self.plot_original_spectrum()
            
            self.status_var.set(f"成功载入文件: {os.path.basename(file_path)}")
            self.log_message(f"成功载入文件: {os.path.basename(file_path)}")
            self.log_message(f"包含 {len(spectra)} 条光谱数据")
            
        except Exception as e:
            messagebox.showerror("错误", f"文件载入失败:\n{str(e)}")
            self.status_var.set("文件载入失败")
            
    def batch_process_folder(self):
        """批量处理文件夹中的光谱文件"""
        folder_path = filedialog.askdirectory(title="选择包含光谱文件的文件夹")
        
        if not folder_path:
            return
            
        try:
            self.status_var.set("正在扫描文件夹...")
            self.root.update()
            
            # 查找支持的文件
            supported_extensions = {'.csv', '.xlsx', '.xls'}
            data_files = []
            
            for filename in os.listdir(folder_path):
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext in supported_extensions:
                    file_path = os.path.join(folder_path, filename)
                    data_files.append(file_path)
            
            if not data_files:
                messagebox.showwarning("警告", "文件夹中没有找到支持的数据文件")
                return
                
            self.status_var.set(f"找到 {len(data_files)} 个文件，开始批量处理...")
            self.root.update()
            
            # 初始化去噪器
            denoiser = SpectralWaveletDenoiser(
                wavelet=self.wavelet_type.get(),
                level=self.decomposition_level.get()
            )
            
            # 批量处理
            batch_results = {}
            successful_files = []
            
            for i, file_path in enumerate(data_files):
                try:
                    self.status_var.set(f"处理进度: {i+1}/{len(data_files)} - {os.path.basename(file_path)}")
                    self.root.update()
                    
                    # 载入单个文件（使用改进的自动识别功能）
                    wavelength, spectra, spectrum_names = load_spectral_data(
                        file_path, None, None  # 让函数自动识别波长列和光谱列
                    )
                    
                    # 对每条光谱进行去噪
                    denoised_spectra = []
                    for spectrum in spectra:
                        denoised_spectrum, threshold = self.denoise_single_spectrum(
                            spectrum, denoiser
                        )
                        denoised_spectra.append(denoised_spectrum)
                    
                    # 存储结果
                    batch_results[os.path.basename(file_path)] = {
                        'wavelength': wavelength,
                        'original_spectra': spectra,
                        'denoised_spectra': np.array(denoised_spectra),
                        'spectrum_names': spectrum_names,
                        'threshold': threshold
                    }
                    successful_files.append(os.path.basename(file_path))
                    
                except Exception as e:
                    self.log_message(f"处理文件 {os.path.basename(file_path)} 失败: {str(e)}")
                    continue
            
            # 存储批量处理结果
            self.batch_results = batch_results
            
            # 计算批量处理前后的信噪比
            self.calculate_batch_snr_before_after()
            
            self.status_var.set(f"批量处理完成 ({len(successful_files)}/{len(data_files)} 成功)")
            self.log_message(f"\n=== 批量处理完成 ===")
            self.log_message(f"成功处理文件数: {len(successful_files)}")
            self.log_message(f"处理失败文件数: {len(data_files) - len(successful_files)}")
            
            # 显示处理结果
            self.show_batch_processing_summary()
            
        except Exception as e:
            messagebox.showerror("错误", f"批量处理失败:\n{str(e)}")
            self.status_var.set("批量处理失败")
            
    def merge_processed_files(self):
        """合并批量处理前后的文件为一个文件"""
        if not self.batch_results:
            messagebox.showwarning("警告", "没有批量处理结果可供合并")
            return
            
        try:
            # 选择保存路径
            save_path = filedialog.asksaveasfilename(
                title="保存合并结果文件",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
            )
            
            if not save_path:
                return
                
            self.status_var.set("正在合并文件...")
            self.root.update()
            
            # 合并所有数据
            merged_data = []
            all_filenames = list(self.batch_results.keys())
            
            # 使用第一个文件的波长作为基准
            base_wavelength = self.batch_results[all_filenames[0]]['wavelength']
            
            # 创建合并的DataFrame
            merged_df = pd.DataFrame({'wavelength': base_wavelength})
            
            # 添加原始数据和去噪数据
            for filename in all_filenames:
                result = self.batch_results[filename]
                spectrum_names = result['spectrum_names']
                
                for i, name in enumerate(spectrum_names):
                    # 原始数据列名
                    original_col = f"{filename}_{name}_original"
                    merged_df[original_col] = result['original_spectra'][i]
                    
                    # 去噪数据列名
                    denoised_col = f"{filename}_{name}_denoised"
                    merged_df[denoised_col] = result['denoised_spectra'][i]
            
            # 保存文件
            if save_path.endswith('.csv'):
                merged_df.to_csv(save_path, index=False)
            else:
                merged_df.to_excel(save_path, index=False)
            
            self.status_var.set(f"合并文件已保存: {os.path.basename(save_path)}")
            self.log_message(f"合并结果已保存至: {save_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"合并文件失败:\n{str(e)}")
            self.status_var.set("合并文件失败")
            
    def save_results(self):
        """保存当前处理结果"""
        if self.denoised_data is None and not self.batch_results:
            messagebox.showwarning("警告", "没有处理结果可供保存")
            return
            
        try:
            # 选择保存路径
            save_path = filedialog.asksaveasfilename(
                title="保存处理结果",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
            )
            
            if not save_path:
                return
                
            self.status_var.set("正在保存结果...")
            self.root.update()
            
            if self.denoised_data is not None:
                # 保存单个文件结果
                df = pd.DataFrame({'wavelength': self.denoised_data['wavelength']})
                for i, name in enumerate(self.denoised_data['spectrum_names']):
                    df[f'{name}_original'] = self.original_data['spectra'][i]
                    df[f'{name}_denoised'] = self.denoised_data['spectra'][i]
                
                if save_path.endswith('.csv'):
                    df.to_csv(save_path, index=False)
                else:
                    df.to_excel(save_path, index=False)
                    
            elif self.batch_results:
                # 保存批量处理摘要
                summary_data = []
                for filename, result in self.batch_results.items():
                    summary_data.append({
                        'filename': filename,
                        'spectra_count': len(result['spectrum_names']),
                        'wavelength_points': len(result['wavelength']),
                        'threshold_used': result['threshold']
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_path = save_path.replace('.csv', '_summary.csv').replace('.xlsx', '_summary.xlsx')
                
                if summary_path.endswith('.csv'):
                    summary_df.to_csv(summary_path, index=False)
                else:
                    summary_df.to_excel(summary_path, index=False)
                
                self.log_message(f"批量处理摘要已保存至: {summary_path}")
            
            self.status_var.set(f"结果已保存: {os.path.basename(save_path)}")
            self.log_message(f"处理结果已保存至: {save_path}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存结果失败:\n{str(e)}")
            self.status_var.set("保存结果失败")
            
    # ==================== 处理功能 ====================
    
    def perform_wavelet_denoise(self):
        """执行小波去噪处理"""
        if self.original_data is None:
            messagebox.showwarning("警告", "请先载入数据文件")
            return
            
        try:
            self.status_var.set("正在执行小波去噪...")
            self.root.update()
            
            # 初始化去噪器
            denoiser = SpectralWaveletDenoiser(
                wavelet=self.wavelet_type.get(),
                level=self.decomposition_level.get()
            )
            
            # 对所有光谱进行去噪
            denoised_spectra = []
            thresholds = []
            
            for i, spectrum in enumerate(self.original_data['spectra']):
                denoised_spectrum, threshold = self.denoise_single_spectrum(
                    spectrum, denoiser
                )
                denoised_spectra.append(denoised_spectrum)
                thresholds.append(threshold)
            
            # 存储去噪结果
            self.denoised_data = {
                'wavelength': self.original_data['wavelength'],
                'spectra': np.array(denoised_spectra),
                'spectrum_names': self.original_data['spectrum_names'],
                'thresholds': thresholds
            }
            
            self.status_var.set("小波去噪完成")
            self.log_message(f"\n=== 小波去噪完成 ===")
            self.log_message(f"使用小波: {self.wavelet_type.get()}")
            self.log_message(f"分解层数: {self.decomposition_level.get()}")
            self.log_message(f"平均阈值: {np.mean(thresholds):.6f}")
            
            # 显示结果
            self.plot_spectrum_comparison()
            
        except Exception as e:
            messagebox.showerror("错误", f"小波去噪失败:\n{str(e)}")
            self.status_var.set("小波去噪失败")
            
    def calculate_snr_analysis(self):
        """计算信噪比分析"""
        if self.original_data is None or self.denoised_data is None:
            messagebox.showwarning("警告", "请先执行小波去噪处理")
            return
            
        try:
            self.status_var.set("正在计算信噪比...")
            self.root.update()
            
            # 计算每个波长点的信噪比
            # SNR_i = μ_i / σ_i （μ为均值，σ为标准差）
            original_spectra = self.original_data['spectra']
            denoised_spectra = self.denoised_data['spectra']
            
            # 计算噪声（原始数据与去噪数据的差异）
            noise_spectra = original_spectra - denoised_spectra
            
            # 对每个波长点计算统计量
            wavelength_points = len(self.original_data['wavelength'])
            snr_values = np.zeros(wavelength_points)
            
            for i in range(wavelength_points):
                # 提取该波长点的所有光谱值
                original_vals = original_spectra[:, i]
                denoised_vals = denoised_spectra[:, i]
                noise_vals = noise_spectra[:, i]
                
                # 计算信噪比：SNR = signal_mean / noise_std
                signal_mean = np.mean(denoised_vals)
                noise_std = np.std(noise_vals)
                
                if noise_std > 1e-10:  # 避免除零
                    snr_values[i] = signal_mean / noise_std
                else:
                    snr_values[i] = 1e6  # 设置大值
            
            # 存储SNR结果
            self.snr_data = {
                'wavelength': self.original_data['wavelength'],
                'snr_values': snr_values,
                'original_snr_stats': self.calculate_snr_statistics(original_spectra),
                'denoised_snr_stats': self.calculate_snr_statistics(denoised_spectra)
            }
            
            self.status_var.set("信噪比计算完成")
            self.log_message(f"\n=== 信噪比分析结果 ===")
            self.log_message(f"平均信噪比: {np.mean(snr_values):.2f}")
            self.log_message(f"最大信噪比: {np.max(snr_values):.2f}")
            self.log_message(f"最小信噪比: {np.min(snr_values):.2f}")
            
            # 显示SNR图
            self.plot_snr_spectrum()
            
        except Exception as e:
            messagebox.showerror("错误", f"信噪比计算失败:\n{str(e)}")
            self.status_var.set("信噪比计算失败")
            
    def show_comparison_plots(self):
        """显示对比图"""
        if self.original_data is None:
            messagebox.showwarning("警告", "请先载入数据")
            return
            
        # 显示光谱对比
        self.plot_spectrum_comparison()
        
        # 如果有SNR数据，也显示SNR对比
        if hasattr(self, 'snr_data'):
            self.plot_snr_spectrum()
            
    # ==================== 辅助方法 ====================
    
    def denoise_single_spectrum(self, spectrum, denoiser):
        """对单条光谱进行去噪"""
        return denoiser.denoise_single_spectrum(spectrum)
        
    def calculate_batch_snr_statistics(self):
        """计算批量处理的整体SNR统计信息"""
        if not self.batch_results:
            return None
            
        # 收集所有文件的数据
        all_filenames = list(self.batch_results.keys())
        base_wavelength = self.batch_results[all_filenames[0]]['wavelength']
        wavelength_points = len(base_wavelength)
        
        # 收集所有文件中每个波长点的信号值
        all_signal_values = []
        
        for i in range(wavelength_points):
            wavelength_signals = []
            for filename in all_filenames:
                result = self.batch_results[filename]
                # 使用去噪后的数据作为信号
                for spectrum in result['denoised_spectra']:
                    wavelength_signals.append(spectrum[i])
            all_signal_values.append(wavelength_signals)
        
        # 计算每个波长点的SNR
        snr_values = np.zeros(wavelength_points)
        for i in range(wavelength_points):
            signals = np.array(all_signal_values[i])
            if len(signals) > 1:
                mean_signal = np.mean(signals)
                std_signal = np.std(signals, ddof=1)
                if std_signal > 1e-10:
                    snr_values[i] = mean_signal / std_signal
                else:
                    snr_values[i] = 1e6
            else:
                snr_values[i] = 0
        
        # 计算统计信息（排除异常值）
        valid_snr = snr_values[snr_values < 1e6]
        if len(valid_snr) > 0:
            stats = {
                'mean': np.mean(valid_snr),
                'std': np.std(valid_snr),
                'max': np.max(valid_snr),
                'min': np.min(valid_snr),
                'median': np.median(valid_snr),
                'valid_points': len(valid_snr),
                'total_points': wavelength_points
            }
        else:
            stats = {
                'mean': 0, 'std': 0, 'max': 0, 'min': 0, 'median': 0,
                'valid_points': 0, 'total_points': wavelength_points
            }
        
        return {
            'wavelength': base_wavelength,
            'snr_values': snr_values,
            'statistics': stats
        }
    def calculate_snr_statistics(self, spectra):
        """计算光谱数据的SNR统计信息"""
        # 对于多条光谱，计算帧间信噪比
        if len(spectra) > 1:
            mean_spectrum = np.mean(spectra, axis=0)
            std_spectrum = np.std(spectra, axis=0)
            snr_spectrum = np.where(std_spectrum > 0, mean_spectrum / std_spectrum, 0)
            return {
                'mean': np.mean(snr_spectrum),
                'std': np.std(snr_spectrum),
                'max': np.max(snr_spectrum),
                'min': np.min(snr_spectrum)
            }
        else:
            return {'mean': 0, 'std': 0, 'max': 0, 'min': 0}
            
    def update_file_info(self):
        """更新文件信息显示"""
        if self.original_data is None:
            return
            
        info_text = f"文件: {self.current_file_path}\n"
        info_text += f"波长点数: {len(self.original_data['wavelength'])}\n"
        info_text += f"光谱条数: {len(self.original_data['spectra'])}\n"
        info_text += f"波长范围: {self.original_data['wavelength'][0]:.1f} - {self.original_data['wavelength'][-1]:.1f} nm\n"
        info_text += f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info_text)
        
    def log_message(self, message):
        """在结果区域记录消息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}\n"
        self.result_text.insert(tk.END, log_entry)
        self.result_text.see(tk.END)
        
    def show_batch_processing_summary(self):
        """显示批量处理摘要"""
        if not self.batch_results:
            return
            
        self.log_message(f"\n=== 批量处理摘要 ===")
        for filename, result in self.batch_results.items():
            self.log_message(f"文件: {filename}")
            self.log_message(f"  光谱数: {len(result['spectrum_names'])}")
            self.log_message(f"  阈值: {result['threshold']:.6f}")
            self.log_message(f"  ---")
            
    def clear_all(self):
        """清除所有数据和显示"""
        self.original_data = None
        self.denoised_data = None
        self.batch_results = {}
        self.current_file_path = None
        
        # 清除显示
        self.info_text.delete(1.0, tk.END)
        self.result_text.delete(1.0, tk.END)
        
        # 清除图形
        self.spectrum_fig.clear()
        self.snr_fig.clear()
        self.spectrum_canvas.draw()
        self.snr_canvas.draw()
        
        self.status_var.set("已清除所有数据")
        self.log_message("所有数据已清除")
        
    def export_results_to_file(self):
        """导出结果到文件"""
        if not self.result_text.get(1.0, tk.END).strip():
            messagebox.showwarning("警告", "没有结果数据可供导出")
            return
            
        try:
            save_path = filedialog.asksaveasfilename(
                title="导出结果",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(self.result_text.get(1.0, tk.END))
                self.log_message(f"结果已导出至: {save_path}")
                
        except Exception as e:
            messagebox.showerror("错误", f"导出失败:\n{str(e)}")

# ==================== 绘图方法 ====================

    def plot_original_spectrum(self):
        """绘制原始光谱图"""
        if self.original_data is None:
            return
            
        self.spectrum_fig.clear()
        ax = self.spectrum_fig.add_subplot(111)
        
        wavelength = self.original_data['wavelength']
        spectra = self.original_data['spectra']
        names = self.original_data['spectrum_names']
        
        for i, (spectrum, name) in enumerate(zip(spectra, names)):
            ax.plot(wavelength, spectrum, linewidth=1, alpha=0.7, 
                   label=f'{name}')
        
        ax.set_xlabel('波长 (nm)')
        ax.set_ylabel('强度值')
        ax.set_title('原始光谱数据')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.spectrum_fig.tight_layout()
        self.spectrum_canvas.draw()
        
    def plot_denoised_spectrum(self):
        """绘制去噪光谱图"""
        if self.denoised_data is None:
            return
            
        self.spectrum_fig.clear()
        ax = self.spectrum_fig.add_subplot(111)
        
        wavelength = self.denoised_data['wavelength']
        spectra = self.denoised_data['spectra']
        names = self.denoised_data['spectrum_names']
        
        for i, (spectrum, name) in enumerate(zip(spectra, names)):
            ax.plot(wavelength, spectrum, linewidth=1.5, 
                   label=f'{name} (去噪)')
        
        ax.set_xlabel('波长 (nm)')
        ax.set_ylabel('强度值')
        ax.set_title('小波去噪后光谱')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.spectrum_fig.tight_layout()
        self.spectrum_canvas.draw()
        
    def plot_spectrum_comparison(self):
        """绘制光谱对比图"""
        if self.original_data is None or self.denoised_data is None:
            return
            
        self.spectrum_fig.clear()
        
        wavelength = self.original_data['wavelength']
        original_spectra = self.original_data['spectra']
        denoised_spectra = self.denoised_data['spectra']
        names = self.original_data['spectrum_names']
        
        # 创建子图
        if len(names) == 1:
            # 单条光谱：上下对比
            ax1 = self.spectrum_fig.add_subplot(211)
            ax2 = self.spectrum_fig.add_subplot(212)
            
            # 上图：原始vs去噪
            ax1.plot(wavelength, original_spectra[0], 'r-', alpha=0.7, 
                    linewidth=1, label='原始光谱')
            ax1.plot(wavelength, denoised_spectra[0], 'b-', linewidth=1.5, 
                    label='去噪光谱')
            ax1.set_ylabel('强度值')
            ax1.set_title('光谱去噪前后对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 下图：差值
            difference = denoised_spectra[0] - original_spectra[0]
            ax2.plot(wavelength, difference, 'g-', linewidth=1)
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('波长 (nm)')
            ax2.set_ylabel('差值')
            ax2.set_title('去噪差值')
            ax2.grid(True, alpha=0.3)
            
        else:
            # 多条光谱：只显示第一条的对比
            ax = self.spectrum_fig.add_subplot(111)
            ax.plot(wavelength, original_spectra[0], 'r-', alpha=0.7, 
                   linewidth=1, label=f'{names[0]} (原始)')
            ax.plot(wavelength, denoised_spectra[0], 'b-', linewidth=1.5, 
                   label=f'{names[0]} (去噪)')
            ax.set_xlabel('波长 (nm)')
            ax.set_ylabel('强度值')
            ax.set_title('第一条光谱去噪前后对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        self.spectrum_fig.tight_layout()
        self.spectrum_canvas.draw()
        
    def plot_snr_spectrum(self):
        """绘制信噪比-波长图"""
        if not hasattr(self, 'snr_data'):
            return
            
        self.snr_fig.clear()
        ax = self.snr_fig.add_subplot(111)
        
        wavelength = self.snr_data['wavelength']
        snr_values = self.snr_data['snr_values']
        
        ax.plot(wavelength, snr_values, 'b-', linewidth=1.5)
        ax.set_xlabel('波长 (nm)')
        ax.set_ylabel('信噪比 SNR')
        ax.set_title('信噪比-波长分布')
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        mean_snr = np.mean(snr_values)
        ax.axhline(y=mean_snr, color='r', linestyle='--', alpha=0.7, 
                  label=f'平均SNR: {mean_snr:.2f}')
        ax.legend()
        
        self.snr_fig.tight_layout()
        self.snr_canvas.draw()
        
    def plot_batch_snr_before_after(self):
        """绘制批量处理前后的SNR对比图"""
        if not hasattr(self, 'batch_snr_results'):
            messagebox.showwarning("警告", "请先进行批量处理并计算信噪比")
            return
            
        self.snr_fig.clear()
        ax = self.snr_fig.add_subplot(111)
        
        wavelength = self.batch_snr_results['wavelength']
        snr_before = self.batch_snr_results['snr_before']
        snr_after = self.batch_snr_results['snr_after']
        
        # 绘制处理前后的SNR曲线
        ax.plot(wavelength, snr_before, 'r-', linewidth=1.5, alpha=0.7, 
                label=f'处理前 (平均: {self.batch_snr_results["statistics"]["before"]["mean"]:.2f})')
        ax.plot(wavelength, snr_after, 'b-', linewidth=1.5, alpha=0.7,
                label=f'处理后 (平均: {self.batch_snr_results["statistics"]["after"]["mean"]:.2f})')
        
        # 添加统计线
        mean_before = self.batch_snr_results['statistics']['before']['mean']
        mean_after = self.batch_snr_results['statistics']['after']['mean']
        ax.axhline(y=mean_before, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=mean_after, color='b', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('波长 (nm)')
        ax.set_ylabel('信噪比 SNR')
        ax.set_title('批量处理前后信噪比对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.snr_fig.tight_layout()
        self.snr_canvas.draw()
        
        self.log_message(f"已显示批量处理前后SNR对比图")
        """绘制批量处理的SNR对比图（正确计算方式）"""
        if not self.batch_results:
            return
            
        self.snr_fig.clear()
        ax = self.snr_fig.add_subplot(111)
        
        # 收集所有文件的数据用于正确计算SNR
        all_filenames = list(self.batch_results.keys())
        base_wavelength = self.batch_results[all_filenames[0]]['wavelength']
        wavelength_points = len(base_wavelength)
        
        # 收集所有文件中每个波长点的信号值
        all_signal_values = []  # 每个元素是一个波长点的所有文件信号值
        
        for i in range(wavelength_points):
            wavelength_signals = []
            for filename in all_filenames:
                result = self.batch_results[filename]
                # 使用去噪后的数据作为信号
                for spectrum in result['denoised_spectra']:
                    wavelength_signals.append(spectrum[i])
            all_signal_values.append(wavelength_signals)
        
        # 计算每个波长点的SNR：SNR_i = μ_i / σ_i
        snr_values = np.zeros(wavelength_points)
        for i in range(wavelength_points):
            signals = np.array(all_signal_values[i])
            if len(signals) > 1:  # 至少需要2个数据点才能计算标准差
                mean_signal = np.mean(signals)
                std_signal = np.std(signals, ddof=1)  # 样本标准差
                if std_signal > 1e-10:
                    snr_values[i] = mean_signal / std_signal
                else:
                    snr_values[i] = 1e6  # 避免除零
            else:
                snr_values[i] = 0  # 数据不足
        
        # 绘制整体SNR曲线
        ax.plot(base_wavelength, snr_values, 'b-', linewidth=2, label='整体SNR')
        
        # 添加统计信息
        mean_snr = np.mean(snr_values[snr_values < 1e6])  # 排除异常值
        ax.axhline(y=mean_snr, color='r', linestyle='--', alpha=0.7, 
                  label=f'平均SNR: {mean_snr:.2f}')
        
        ax.set_xlabel('波长 (nm)')
        ax.set_ylabel('信噪比 SNR')
        ax.set_title('批量处理整体信噪比分析')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.snr_fig.tight_layout()
        self.snr_canvas.draw()

def main():
    """主函数"""
    root = tk.Tk()
    app = SpectralWaveletGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
