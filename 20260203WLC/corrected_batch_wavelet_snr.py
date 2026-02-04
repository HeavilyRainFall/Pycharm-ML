import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QTextEdit, QProgressBar, QComboBox, QSpinBox, 
                            QMessageBox, QTableWidget, QTableWidgetItem, 
                            QTabWidget, QGroupBox, QFormLayout, QSplitter,
                            QCheckBox, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import os
import warnings
import numpy as np
warnings.filterwarnings('ignore')

# 设置matplotlib中文字体 - 更全面的配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Bitstream Vera Sans', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica', 'Avant Garde', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

class BatchWaveletSNRAnalyzer:
    """批量小波去噪与信噪比分析器 - 按照约定方式实现"""
    
    def __init__(self):
        self.original_spectra_data = {}  # {filename: (wavelength, intensity)}
        self.denoised_spectra_data = {}  # {filename: (wavelength, intensity)}
        self.wavelength_grid = None      # 统一波长网格
        self.interpolated_original = None  # 插值后的原始数据矩阵
        self.interpolated_denoised = None  # 插值后的去噪数据矩阵
        
    def load_batch_spectral_files(self, file_paths):
        """批量加载光谱文件（智能处理单文件多列和多文件场景）"""
        print("正在加载批量光谱文件...")
        
        all_virtual_files = []  # 存储虚拟文件信息
        
        # 读取所有文件并智能处理
        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)
                
                # 根据文件扩展名选择正确的读取方法
                if file_path.endswith('.csv'):
                    df = self._load_csv_with_header_detection(file_path)
                else:
                    df = pd.read_excel(file_path)
                
                print(f"  处理文件 {filename}: {df.shape[1]} 列数据")
                
                # 智能判断处理方式
                if df.shape[1] == 2:
                    # 标准两列文件：波长 + 数据，直接处理为单个光谱
                    print(f"    → 标准两列文件，直接处理为单个光谱")
                    wavelength = df.iloc[:, 0].values
                    intensity = df.iloc[:, 1].values
                    
                    virtual_file_data = {
                        'filename': filename,
                        'wavelength': wavelength,
                        'intensity': intensity,
                        'source_file': filename,
                        'column_index': 1
                    }
                    
                    all_virtual_files.append(virtual_file_data)
                    print(f"  ✓ 加载光谱 {filename}: {len(wavelength)} 个数据点")
                    
                elif df.shape[1] >= 3:
                    # 多列文件：波长 + 多个数据列，进行虚拟拆分
                    print(f"    → 多列文件({df.shape[1]}列)，进行虚拟拆分")
                    wavelength = df.iloc[:, 0].values
                    column_names = list(df.columns)
                    
                    # 为每个数据列创建虚拟文件
                    for col_idx in range(1, df.shape[1]):
                        intensity = df.iloc[:, col_idx].values
                        
                        # 生成虚拟文件名
                        if len(column_names) > col_idx and column_names[col_idx]:
                            # 有表头的情况
                            virtual_filename = f"{filename}_{column_names[col_idx]}"
                        else:
                            # 无表头的情况
                            virtual_filename = f"{filename}_col{col_idx+1}"
                        
                        # 创建虚拟文件数据结构
                        virtual_file_data = {
                            'filename': virtual_filename,
                            'wavelength': wavelength,
                            'intensity': intensity,
                            'source_file': filename,
                            'column_index': col_idx
                        }
                        
                        all_virtual_files.append(virtual_file_data)
                        print(f"  ✓ 虚拟拆分 {virtual_filename}: {len(wavelength)} 个数据点")
                else:
                    print(f"  ⚠ {filename}: 数据列数不足，需要至少2列（波长+数据）")
                    continue
                
            except Exception as e:
                print(f"  ✗ 处理 {file_path} 失败: {e}")
                continue
        
        if not all_virtual_files:
            raise ValueError("没有成功加载任何光谱数据")
        
        # 创建统一的波长网格
        min_wavelength = max([vf['wavelength'].min() for vf in all_virtual_files])
        max_wavelength = min([vf['wavelength'].max() for vf in all_virtual_files])
        common_length = min([len(vf['wavelength']) for vf in all_virtual_files])
        
        self.wavelength_grid = np.linspace(min_wavelength, max_wavelength, common_length)
        
        # 转换为原始数据格式（保持接口兼容性）
        self.original_spectra_data = {}
        for vf in all_virtual_files:
            self.original_spectra_data[vf['filename']] = (vf['wavelength'], vf['intensity'])
        
        print(f"创建统一波长网格: {len(self.wavelength_grid)} 点")
        print(f"波长范围: {min_wavelength:.1f} - {max_wavelength:.1f} nm")
        print(f"总共虚拟光谱数: {len(all_virtual_files)} 条")
        print(f"来自 {len(set(vf['source_file'] for vf in all_virtual_files))} 个源文件")
        
        return [vf['filename'] for vf in all_virtual_files]
    
    def _load_csv_with_header_detection(self, file_path):
        """带表头检测的CSV文件加载（遵循项目规范）"""
        # 先尝试读取前几行来判断是否有表头
        try:
            # 读取前3行用于表头检测
            sample_df = pd.read_csv(file_path, nrows=3)
            
            # 检测表头的多种策略
            has_header = self._detect_header_multi_strategy(sample_df)
            
            if has_header:
                # 有表头，正常读取
                df = pd.read_csv(file_path)
                print(f"    检测到表头: {list(df.columns)[:5]}...")
            else:
                # 无表头，指定header=None
                df = pd.read_csv(file_path, header=None)
                # 为无表头数据设置默认列名
                default_columns = ['Wavelength'] + [f'Column_{i}' for i in range(1, df.shape[1])] 
                df.columns = default_columns[:df.shape[1]]
                print(f"    无表头，使用默认列名: {list(df.columns)[:5]}...")
            
            # 数据清洗和验证
            df = self._clean_spectral_data(df)
            return df
            
        except Exception as e:
            print(f"    CSV读取异常: {e}")
            # 最后尝试使用默认方式
            df = pd.read_csv(file_path)
            df = self._clean_spectral_data(df)
            return df
    
    def _detect_header_multi_strategy(self, df_sample):
        """多策略表头检测（结合经验教训）"""
        first_row = df_sample.iloc[0]
        
        # 策略1: 检查语义关键词
        first_row_str = ' '.join(str(item).lower() for item in first_row)
        header_keywords = ['wavelength', 'nm', 'lambda', '波长', 'value', 'intensity', 'power', 'counts']
        keyword_matches = [kw for kw in header_keywords if kw in first_row_str]
        
        if keyword_matches:
            print(f"    关键词检测: 发现 {keyword_matches}")
            return True
        
        # 策略2: 分析数值型占比
        numeric_count = sum(pd.to_numeric(first_row, errors='coerce').notna())
        numeric_ratio = numeric_count / len(first_row)
        
        print(f"    数值占比分析: {numeric_ratio:.2%} ({numeric_count}/{len(first_row)})")
        
        # 策略3: 结合列名多样性
        unique_types = len(set(type(item) for item in first_row))
        
        # 综合判断：如果数值占比<80%且类型多样，则认为是有表头
        has_header = numeric_ratio < 0.8 and unique_types > 1
        
        return has_header
    
    def _clean_spectral_data(self, df):
        """光谱数据清洗和类型转换"""
        # 确保所有列都能转换为数值
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                print(f"    警告: 列 '{col}' 无法完全转换为数值")
        
        # 删除包含NaN的行
        original_rows = len(df)
        df = df.dropna()
        if len(df) < original_rows:
            print(f"    数据清洗: 删除 {original_rows - len(df)} 行含NaN数据")
        
        # 验证数据合理性
        if df.empty:
            raise ValueError("清洗后数据为空")
        
        if df.shape[1] < 2:
            raise ValueError("数据列数不足，至少需要波长和一个测量值列")
        
        return df
    
    def interpolate_to_common_grid(self):
        """将所有光谱插值到统一波长网格"""
        print("正在进行数据插值...")
        
        n_files = len(self.original_spectra_data)
        n_wavelengths = len(self.wavelength_grid)
        
        # 创建插值矩阵
        self.interpolated_original = np.zeros((n_files, n_wavelengths))
        
        for i, (filename, (wavelength, intensity)) in enumerate(self.original_spectra_data.items()):
            # 线性插值到统一波长网格
            interpolated_intensity = np.interp(self.wavelength_grid, wavelength, intensity)
            self.interpolated_original[i, :] = interpolated_intensity
            
        print(f"插值完成: {n_files} 个文件 × {n_wavelengths} 个波长点")
    
    def batch_wavelet_denoise(self, wavelet='db4', level=5, threshold_type='soft', apply_layers=None):
        """批量小波去噪（增强版）
        
        参数:
        wavelet: 小波基函数
        level: 分解层数
        threshold_type: 阈值类型 ('soft' 或 'hard')
        apply_layers: 应用阈值的层数列表
        """
        print(f"开始批量小波去噪")
        print(f"  小波基: {wavelet}")
        print(f"  分解层数: {level}")
        print(f"  阈值类型: {'软阈值' if threshold_type == 'soft' else '硬阈值'}")
        if apply_layers:
            layer_names = [f'D{layer}' for layer in apply_layers]
            print(f"  应用层数: {', '.join(layer_names)}")
        else:
            print(f"  应用层数: 全部细节系数层")
        
        if self.interpolated_original is None:
            raise ValueError("请先进行数据插值")
        
        n_files, n_wavelengths = self.interpolated_original.shape
        self.interpolated_denoised = np.zeros((n_files, n_wavelengths))
        
        # 对每个光谱进行小波去噪
        for i in range(n_files):
            spectrum = self.interpolated_original[i, :]
            denoised_spectrum = self.wavelet_denoise_single(spectrum, wavelet, level, threshold_type, apply_layers)
            self.interpolated_denoised[i, :] = denoised_spectrum
            
            filename = list(self.original_spectra_data.keys())[i]
            print(f"  ✓ 处理 {filename}")
        
        print("批量小波去噪完成")
    
    def wavelet_denoise_single(self, spectrum, wavelet='db4', level=5, threshold_type='soft', apply_layers=None):
        """单条光谱的小波去噪（增强版）
        
        参数:
        spectrum: 输入光谱数据
        wavelet: 小波基函数
        level: 分解层数
        threshold_type: 阈值类型 ('soft' 或 'hard')
        apply_layers: 应用阈值的层数列表，None表示全部应用
        """
        # 小波分解
        coeffs = pywt.wavedec(spectrum, wavelet, level=level)
        
        # 计算阈值
        threshold = self.universal_threshold(coeffs)
        
        # 确定应用阈值的层数
        if apply_layers is None:
            apply_layers = list(range(1, len(coeffs)))  # 默认应用所有细节系数层
        
        # 对细节系数进行阈值处理
        coeffs_thresh = [coeffs[0]]  # 近似系数(A)不变
        
        for i in range(1, len(coeffs)):  # 从D1到Dn
            if i in apply_layers:
                # 应用阈值处理
                if threshold_type == 'soft':
                    processed_coeff = pywt.threshold(coeffs[i], threshold, mode='soft')
                else:  # hard
                    processed_coeff = pywt.threshold(coeffs[i], threshold, mode='hard')
            else:
                # 不应用阈值，保持原系数
                processed_coeff = coeffs[i]
            
            coeffs_thresh.append(processed_coeff)
            
            # 输出调试信息
            layer_name = f'D{i}' if i > 0 else 'A'
            apply_status = '✓应用' if i in apply_layers else '○跳过'
            print(f"    {layer_name}层: 阈值={threshold:.4f}, 状态={apply_status}")
        
        # 重构
        denoised = pywt.waverec(coeffs_thresh, wavelet)
        return denoised[:len(spectrum)]
    
    def universal_threshold(self, coeffs):
        """通用阈值计算"""
        # 使用最后一层细节系数计算阈值
        last_detail = coeffs[-1]
        sigma = np.median(np.abs(last_detail)) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(last_detail)))
        return threshold
    
    def calculate_batch_snr_before_after(self):
        """按照项目规范计算批量处理前后的信噪比：SNR_i = μ_i / σ_i"""
        print("正在计算批量信噪比...")
        
        if self.interpolated_original is None or self.interpolated_denoised is None:
            raise ValueError("请先完成数据处理")
        
        # 按照项目规范：SNR_i = μ_i / σ_i
        # μ_i: 第i个波长点在所有文件中的信号均值
        # σ_i: 第i个波长点在所有文件中的信号标准差
        
        n_wavelengths = self.interpolated_original.shape[1]
        original_snr = np.zeros(n_wavelengths)
        denoised_snr = np.zeros(n_wavelengths)
        
        # 对每个波长点分别计算SNR
        for i in range(n_wavelengths):
            # 收集该波长点在所有文件中的信号值
            original_signals = self.interpolated_original[:, i]  # 所有文件在第i个波长点的值
            denoised_signals = self.interpolated_denoised[:, i]  # 去噪后对应位置的值
            
            # 计算原始数据的SNR
            orig_mean = np.mean(original_signals)
            orig_std = np.std(original_signals, ddof=1)
            if orig_std > 1e-10:
                original_snr[i] = orig_mean / orig_std
            else:
                original_snr[i] = 1e6  # 避免除零，设为大值
            
            # 计算去噪数据的SNR
            denoise_mean = np.mean(denoised_signals)
            denoise_std = np.std(denoised_signals, ddof=1)
            if denoise_std > 1e-10:
                denoised_snr[i] = denoise_mean / denoise_std
            else:
                denoised_snr[i] = 1e6
        
        # 计算改善量
        snr_improvement = denoised_snr - original_snr
        
        # 计算统计量用于显示
        original_mean = np.mean(self.interpolated_original, axis=0)
        original_std = np.std(self.interpolated_original, axis=0, ddof=1)
        denoised_mean = np.mean(self.interpolated_denoised, axis=0)
        denoised_std = np.std(self.interpolated_denoised, axis=0, ddof=1)
        
        results = {
            'wavelength': self.wavelength_grid,
            'original_snr': original_snr,
            'denoised_snr': denoised_snr,
            'snr_improvement': snr_improvement,
            'original_mean': original_mean,
            'original_std': original_std,
            'denoised_mean': denoised_mean,
            'denoised_std': denoised_std
        }
        
        # 输出统计信息
        valid_indices = (original_snr < 1e6) & (denoised_snr < 1e6)
        if np.sum(valid_indices) > 0:
            avg_original_snr = np.mean(original_snr[valid_indices])
            avg_denoised_snr = np.mean(denoised_snr[valid_indices])
            avg_improvement = np.mean(snr_improvement[valid_indices])
            
            print(f"处理前平均SNR: {avg_original_snr:.2f}")
            print(f"处理后平均SNR: {avg_denoised_snr:.2f}")
            print(f"平均SNR改善: {avg_improvement:+.2f}")
            print(f"SNR改善比例: {(avg_denoised_snr/avg_original_snr - 1)*100:+.1f}%")
        else:
            print("警告：大部分SNR值异常，可能数据存在问题")
        
        return results

class BatchProcessingWorker(QThread):
    """批量处理工作线程"""
    progress_updated = pyqtSignal(int)
    processing_finished = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_paths, wavelet, level, threshold_type, apply_layers):
        super().__init__()
        self.file_paths = file_paths
        self.wavelet = wavelet
        self.level = level
        self.threshold_type = threshold_type
        self.apply_layers = apply_layers
        self.analyzer = BatchWaveletSNRAnalyzer()
        
    def run(self):
        try:
            # 步骤1: 加载文件
            self.progress_updated.emit(10)
            filenames = self.analyzer.load_batch_spectral_files(self.file_paths)
            
            # 步骤2: 数据插值
            self.progress_updated.emit(30)
            self.analyzer.interpolate_to_common_grid()
            
            # 步骤3: 小波去噪
            self.progress_updated.emit(60)
            # 转换阈值类型中文到英文
            threshold_mode = 'soft' if self.threshold_type == '软阈值' else 'hard'
            self.analyzer.batch_wavelet_denoise(self.wavelet, self.level, threshold_mode, self.apply_layers)
            
            # 步骤4: 计算信噪比
            self.progress_updated.emit(80)
            snr_results = self.analyzer.calculate_batch_snr_before_after()
            
            # 步骤5: 完成
            self.progress_updated.emit(100)
            self.processing_finished.emit(snr_results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class CorrectedBatchWaveletGUI(QMainWindow):
    """修正后的批量小波去噪GUI - 符合我们的约定"""
    
    def __init__(self):
        super().__init__()
        self.snr_results = None
        self.init_ui()
        
    def select_all_layers(self):
        """全选所有层数"""
        for checkbox in self.layer_checkboxes:
            checkbox.setChecked(True)
            
    def select_none_layers(self):
        """全不选所有层数"""
        for checkbox in self.layer_checkboxes:
            checkbox.setChecked(False)
        
    def init_ui(self):
        self.setWindowTitle('🔬 光谱小波去噪与信噪比分析系统 Professional Edition v2.1')
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # 创建标题
        title_label = QLabel('🔬 光谱小波去噪与信噪比分析系统')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #2c3e50;
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 5px;
                margin-bottom: 10px;
            }
        """)
        main_layout.addWidget(title_label)
        
        # 创建软件说明区域
        software_info = QTextEdit()
        software_info.setMaximumHeight(180)
        software_info.setReadOnly(True)
        software_info.setStyleSheet("""
            QTextEdit {
                background-color: #fff8dc;
                border: 2px solid #f39c12;
                border-radius: 8px;
                font-family: 'Microsoft YaHei', sans-serif;
                font-size: 12px;
                padding: 10px;
            }
        """)
        software_info.setHtml("""
        <div style='color: #2c3e50;'>
        <h3 style='color: #c0392b; margin-bottom: 10px; border-bottom: 2px solid #f39c12; padding-bottom: 5px;'>
        📋 软件说明与使用指南</h3>
        
        <table style='width: 100%; border-collapse: collapse;'>
        <tr>
            <td style='vertical-align: top; width: 50%; padding-right: 15px;'>
                <h4 style='color: #2980b9; margin-bottom: 8px;'>🎯 软件功能</h4>
                <ul style='margin-top: 5px; margin-left: 20px; line-height: 1.4;'>
                    <li><b>批量处理</b>：支持同时处理多个光谱文件</li>
                    <li><b>小波去噪</b>：采用DB4小波基进行多层分解去噪</li>
                    <li><b>信噪比分析</b>：计算处理前后的SNR变化</li>
                    <li><b>可视化展示</b>：提供光谱对比图和统计分析图</li>
                    <li><b>交互式操作</b>：支持光谱显示切换和图表缩放</li>
                </ul>
                
                <h4 style='color: #27ae60; margin-top: 15px; margin-bottom: 8px;'>📊 输出结果</h4>
                <ul style='margin-top: 5px; margin-left: 20px; line-height: 1.4;'>
                    <li>处理前后光谱对比图</li>
                    <li>SNR改善统计图表</li>
                    <li>详细数值统计表格</li>
                    <li>Excel格式结果导出</li>
                </ul>
            </td>
            
            <td style='vertical-align: top; width: 50%; padding-left: 15px; border-left: 1px dashed #bdc3c7;'>
                <h4 style='color: #8e44ad; margin-bottom: 8px;'>📁 文件载入要求</h4>
                <ul style='margin-top: 5px; margin-left: 20px; line-height: 1.4;'>
                    <li><b>格式支持</b>：CSV、Excel(.xlsx/.xls)文件</li>
                    <li><b>数据结构</b>：第一列为波长(nm)，后续列为光谱强度</li>
                    <li><b>编码要求</b>：UTF-8编码（推荐）或GBK编码</li>
                    <li><b>数据质量</b>：避免空行、特殊字符、非数值数据</li>
                    <li><b>文件大小</b>：单文件建议不超过10MB</li>
                </ul>
                
                <h4 style='color: #e67e22; margin-top: 15px; margin-bottom: 8px;'>⚠️ 注意事项</h4>
                <ul style='margin-top: 5px; margin-left: 20px; line-height: 1.4;'>
                    <li>确保所有文件具有相同的波长范围</li>
                    <li>建议文件名使用英文或数字命名</li>
                    <li>处理大量文件时请耐心等待</li>
                    <li>结果会自动保存在程序运行目录</li>
                </ul>
            </td>
        </tr>
        </table>
        
        <div style='margin-top: 12px; padding: 8px; background-color: #e8f6f3; border-left: 4px solid #1abc9c; border-radius: 3px;'>
        <b>💡 快速开始：</b> 点击左侧"选择文件"按钮导入光谱数据，设置合适参数后点击"开始批量处理"即可。
        </div>
        </div>
        """)
        main_layout.addWidget(software_info)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # 右侧结果显示
        result_display = self.create_result_display()
        splitter.addWidget(result_display)
        
        # 设置分割比例
        splitter.setSizes([400, 800])
        
    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 文件选择区域
        file_group = QGroupBox("📁 文件选择与管理")
        file_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3498db;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        file_layout = QVBoxLayout(file_group)
        
        self.file_list_widget = QTextEdit()
        self.file_list_widget.setMaximumHeight(150)
        self.file_list_widget.setPlaceholderText("请选择CSV或Excel格式的光谱文件...\n支持单文件多光谱列或多个文件批量处理")
        
        btn_layout = QHBoxLayout()
        self.select_files_btn = QPushButton("选择文件")
        self.select_files_btn.clicked.connect(self.select_files)
        self.clear_files_btn = QPushButton("清空列表")
        self.clear_files_btn.clicked.connect(self.clear_files)
        btn_layout.addWidget(self.select_files_btn)
        btn_layout.addWidget(self.clear_files_btn)
        
        file_layout.addLayout(btn_layout)
        file_layout.addWidget(self.file_list_widget)
        
        # 参数设置区域
        param_group = QGroupBox("⚙️ 处理参数设置")
        param_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #27ae60;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        param_layout = QFormLayout(param_group)
        
        # 小波基选择
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(['db4', 'db8', 'sym4', 'sym8', 'coif1', 'coif3'])
        self.wavelet_combo.setCurrentText('db4')
        self.wavelet_combo.setToolTip("选择小波基函数\ndb4: Daubechies 4阶（推荐）\ndb8: Daubechies 8阶\nsym4: Symlets 4阶\nsym8: Symlets 8阶\ncoif1: Coiflets 1阶\ncoif3: Coiflets 3阶")
        
        # 分解层数
        self.level_spin = QSpinBox()
        self.level_spin.setRange(1, 10)
        self.level_spin.setValue(5)
        self.level_spin.setToolTip("小波分解层数\n推荐值：4-6层\n层数越多，频率分辨率越高，但可能过度去噪")
        
        # 阈值类型选择
        self.threshold_type_combo = QComboBox()
        self.threshold_type_combo.addItems(['软阈值', '硬阈值'])
        self.threshold_type_combo.setCurrentText('软阈值')
        self.threshold_type_combo.setToolTip("阈值处理类型：\n软阈值：sign(x) * max(|x| - λ, 0)\n硬阈值：如果|x| > λ则保留x，否则置0\n软阈值通常能更好地保持信号特征")
        
        # 应用层数选择
        self.apply_layers_group = QGroupBox("📈 应用阈值的层数选择")
        self.apply_layers_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #f39c12;
                border-radius: 6px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        apply_layout = QVBoxLayout(self.apply_layers_group)
        
        # 全选/全不选按钮
        select_btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("全选")
        self.select_all_btn.clicked.connect(self.select_all_layers)
        self.select_none_btn = QPushButton("全不选")
        self.select_none_btn.clicked.connect(self.select_none_layers)
        select_btn_layout.addWidget(self.select_all_btn)
        select_btn_layout.addWidget(self.select_none_btn)
        select_btn_layout.addStretch()
        apply_layout.addLayout(select_btn_layout)
        
        # 层数复选框
        self.layer_checkboxes = []
        layers_layout = QGridLayout()
        for i in range(6):
            checkbox = QCheckBox(f"第{i+1}层细节系数 (D{i+1})")
            checkbox.setChecked(True)
            checkbox.setToolTip(f"D{i+1}层对应频率范围约为原始信号的1/{2**(i+1)}")
            self.layer_checkboxes.append(checkbox)
            row = i // 3
            col = i % 3
            layers_layout.addWidget(checkbox, row, col)
        apply_layout.addLayout(layers_layout)
        
        # 参数说明文本
        param_explanation = QTextEdit()
        param_explanation.setMaximumHeight(180)
        param_explanation.setReadOnly(True)
        param_explanation.setStyleSheet("background-color: #f8f9fa; font-size: 11px; border: 1px solid #ddd;")
        param_explanation.setHtml("""
        <div style='font-family: Microsoft YaHei, sans-serif;'>
        <h3 style='color: #2c3e50; margin-bottom: 8px;'>⚙️ 处理参数详解</h3>
        
        <p><b>📊 小波基函数选择：</b></p>
        <ul style='margin-top: 2px; margin-bottom: 8px;'>
            <li><b>db4/db8</b>：Daubechies系列，适合一般信号处理</li>
            <li><b>sym4/sym8</b>：Symlets系列，对称性更好</li>
            <li><b>coif1/coif3</b>：Coiflets系列，平滑性更佳</li>
        </ul>
        
        <p><b>🔍 分解层数：</b></p>
        <ul style='margin-top: 2px; margin-bottom: 8px;'>
            <li>影响频率分辨率和去噪效果</li>
            <li>推荐值：4-6层</li>
            <li>层数过多可能导致过度平滑</li>
        </ul>
        
        <p><b>🎯 阈值处理类型：</b></p>
        <ul style='margin-top: 2px; margin-bottom: 8px;'>
            <li><span style='color: #27ae60;'>● 软阈值</span>：温和收缩，保持信号连续性</li>
            <li><span style='color: #e74c3c;'>● 硬阈值</span>：直接截断，去噪效果更强但可能产生伪影</li>
        </ul>
        
        <p><b>📈 应用层数选择：</b></p>
        <ul style='margin-top: 2px; margin-bottom: 8px;'>
            <li>D1-D3：高频噪声层（通常建议处理）</li>
            <li>D4-D6：中低频信息层（谨慎处理）</li>
            <li>A：近似系数层（一般不处理）</li>
        </ul>
        </div>
        """)
        
        param_layout.addRow("小波基:", self.wavelet_combo)
        param_layout.addRow("分解层数:", self.level_spin)
        param_layout.addRow("阈值类型:", self.threshold_type_combo)
        param_layout.addRow("应用层数:", self.apply_layers_group)
        param_layout.addRow("参数说明:", param_explanation)
        
        # 处理按钮
        self.process_btn = QPushButton("🚀 开始批量处理")
        self.process_btn.clicked.connect(self.start_batch_processing)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #27ae60, stop:1 #2ecc71);
                color: white;
                font-weight: bold;
                padding: 12px;
                font-size: 14px;
                border-radius: 8px;
                border: none;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #219653, stop:1 #27ae60);
            }
            QPushButton:pressed {
                background-color: #219653;
            }
        """)
        
        self.save_btn = QPushButton("💾 保存结果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9);
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 12px;
                border-radius: 6px;
                border: none;
            }
            QPushButton:hover {
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #3498db);
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
        """)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # 状态显示
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        
        # 添加到布局
        layout.addWidget(file_group)
        layout.addWidget(param_group)
        layout.addWidget(self.process_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("处理状态:"))
        layout.addWidget(self.status_text)
        
        return panel
    
    def create_result_display(self):
        """创建结果显示区域"""
        display = QWidget()
        layout = QVBoxLayout(display)
        
        # 创建标签页
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)
        
        # SNR对比图标签页
        self.create_snr_comparison_tab(tab_widget)
        
        # 统计信息标签页
        self.create_statistics_tab(tab_widget)
        
        # 原始数据对比标签页
        self.create_data_comparison_tab(tab_widget)
        
        return display
    
    def create_snr_comparison_tab(self, tab_widget):
        """创建SNR对比图标签页"""
        snr_tab = QWidget()
        layout = QVBoxLayout(snr_tab)
        
        # 创建matplotlib图形
        self.snr_figure, self.snr_axes = plt.subplots(2, 2, figsize=(12, 10))
        self.snr_canvas = FigureCanvas(self.snr_figure)
        
        # 添加导航工具栏（缩放功能）
        self.snr_toolbar = NavigationToolbar(self.snr_canvas, snr_tab)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.plot_snr_comparison_btn = QPushButton("绘制SNR对比图")
        self.plot_snr_comparison_btn.clicked.connect(self.plot_snr_comparison)
        self.plot_improvement_btn = QPushButton("绘制改善图")
        self.plot_improvement_btn.clicked.connect(self.plot_improvement)
        self.plot_statistical_btn = QPushButton("统计分析图")
        self.plot_statistical_btn.clicked.connect(self.plot_statistical_analysis)
        
        btn_layout.addWidget(self.plot_snr_comparison_btn)
        btn_layout.addWidget(self.plot_improvement_btn)
        btn_layout.addWidget(self.plot_statistical_btn)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        layout.addWidget(self.snr_toolbar)
        layout.addWidget(self.snr_canvas)
        tab_widget.addTab(snr_tab, "SNR对比分析")
    
    def create_statistics_tab(self, tab_widget):
        """创建统计信息标签页"""
        stats_tab = QWidget()
        layout = QVBoxLayout(stats_tab)
        
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(5)
        # 设置表头字体为支持中文的字体
        header_labels = ['波长(nm)', '处理前SNR', '处理后SNR', 'SNR改善', '改善比例(%)']
        self.stats_table.setHorizontalHeaderLabels(header_labels)
        # 设置表头字体
        header_font = QFont('Microsoft YaHei', 9)
        self.stats_table.horizontalHeader().setFont(header_font)
        
        layout.addWidget(self.stats_table)
        tab_widget.addTab(stats_tab, "详细统计")
    
    def create_data_comparison_tab(self, tab_widget):
        """创建数据对比标签页"""
        data_tab = QWidget()
        layout = QVBoxLayout(data_tab)
        
        # 创建数据对比图形
        self.data_figure, self.data_axes = plt.subplots(2, 1, figsize=(12, 8))
        self.data_canvas = FigureCanvas(self.data_figure)
        
        # 添加导航工具栏（缩放功能）
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.data_toolbar = NavigationToolbar(self.data_canvas, data_tab)
        
        # 控制按钮
        btn_layout = QHBoxLayout()
        self.plot_spectra_btn = QPushButton("绘制光谱对比")
        self.plot_spectra_btn.clicked.connect(self.plot_spectra_comparison)
        self.plot_noise_btn = QPushButton("绘制噪声分析")
        self.plot_noise_btn.clicked.connect(self.plot_noise_analysis)
        
        # 添加显示选项
        self.show_original_cb = QCheckBox("显示原始光谱")
        self.show_original_cb.setChecked(True)
        self.show_denoised_cb = QCheckBox("显示去噪光谱")
        self.show_denoised_cb.setChecked(True)
        
        btn_layout.addWidget(self.plot_spectra_btn)
        btn_layout.addWidget(self.plot_noise_btn)
        btn_layout.addWidget(self.show_original_cb)
        btn_layout.addWidget(self.show_denoised_cb)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        layout.addWidget(self.data_toolbar)
        layout.addWidget(self.data_canvas)
        tab_widget.addTab(data_tab, "数据对比")
    
    def select_files(self):
        """选择文件"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择光谱数据文件", "", 
            "数据文件 (*.csv *.xlsx *.xls);;所有文件 (*)"
        )
        
        if files:
            current_text = self.file_list_widget.toPlainText()
            for file in files:
                if file not in current_text:
                    current_text += file + '\n'
            self.file_list_widget.setPlainText(current_text)
            
            self.status_text.append(f"已选择 {len(files)} 个文件")
    
    def clear_files(self):
        """清空文件列表"""
        self.file_list_widget.clear()
        self.status_text.clear()
        self.stats_table.setRowCount(0)
        self.save_btn.setEnabled(False)
        self.snr_results = None
        
    def start_batch_processing(self):
        """开始批量处理"""
        file_paths = [line.strip() for line in self.file_list_widget.toPlainText().split('\n') if line.strip()]
        
        if len(file_paths) < 1:
            QMessageBox.warning(self, "警告", "请至少选择1个文件进行处理！")
            return
        
        # 获取参数
        wavelet = self.wavelet_combo.currentText()
        level = self.level_spin.value()
        threshold_type = self.threshold_type_combo.currentText()
        
        # 获取应用层数选择
        apply_layers = []
        for i, checkbox in enumerate(self.layer_checkboxes):
            if checkbox.isChecked():
                apply_layers.append(i + 1)  # D1对应索引1
        
        # 如果没有选择任何层，给出提示
        if not apply_layers:
            reply = QMessageBox.question(self, '确认', 
                                       '您没有选择任何应用层数，这意味着不会对任何细节系数进行阈值处理。\n是否继续？',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return
        
        # 显示处理参数摘要
        layer_names = [f'D{layer}' for layer in apply_layers] if apply_layers else ['无']
        self.status_text.append(f"\n📋 处理参数摘要：")
        self.status_text.append(f"   小波基：{wavelet}")
        self.status_text.append(f"   分解层数：{level}")
        self.status_text.append(f"   阈值类型：{threshold_type}")
        self.status_text.append(f"   应用层数：{', '.join(layer_names)}")
        self.status_text.append(f"   处理文件数：{len(file_paths)}")
        
        # 创建并启动处理线程
        self.worker = BatchProcessingWorker(file_paths, wavelet, level, threshold_type, apply_layers)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.processing_finished.connect(self.processing_completed)
        self.worker.error_occurred.connect(self.processing_error)
        
        # 更新UI状态
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_text.append("\n🚀 开始批量处理...")
        
        self.worker.start()
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        progress_messages = {
            10: "正在加载文件...",
            30: "正在进行数据插值...",
            60: "正在执行小波去噪...",
            80: "正在计算信噪比...",
            100: "处理完成！"
        }
        if value in progress_messages:
            self.status_text.append(progress_messages[value])
    
    def processing_completed(self, results):
        """处理完成"""
        self.snr_results = results
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.save_btn.setEnabled(True)
        
        self.status_text.append("✓ 批量处理完成！")
        self.status_text.append(f"处理文件数: {len(self.snr_results['original_snr'])}")
        self.status_text.append(f"波长点数: {len(self.snr_results['wavelength'])}")
        
        # 更新统计表格
        self.update_statistics_table()
        
        # 自动绘制初始图表
        self.plot_snr_comparison()
        
        QMessageBox.information(self, "完成", "批量处理完成！请查看结果分析。")
    
    def processing_error(self, error_message):
        """处理错误"""
        self.process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_text.append(f"✗ 处理失败: {error_message}")
        QMessageBox.critical(self, "错误", f"处理过程中发生错误:\n{error_message}")
    
    def update_statistics_table(self):
        """更新统计表格"""
        if not self.snr_results:
            return
            
        wavelength = self.snr_results['wavelength']
        original_snr = self.snr_results['original_snr']
        denoised_snr = self.snr_results['denoised_snr']
        improvement = self.snr_results['snr_improvement']
        
        # 只显示前50个波长点（避免表格过大）
        display_count = min(50, len(wavelength))
        
        self.stats_table.setRowCount(display_count)
        
        for i in range(display_count):
            self.stats_table.setItem(i, 0, QTableWidgetItem(f"{wavelength[i]:.1f}"))
            self.stats_table.setItem(i, 1, QTableWidgetItem(f"{original_snr[i]:.2f}"))
            self.stats_table.setItem(i, 2, QTableWidgetItem(f"{denoised_snr[i]:.2f}"))
            self.stats_table.setItem(i, 3, QTableWidgetItem(f"{improvement[i]:+.2f}"))
            if original_snr[i] > 1e-10:
                improvement_pct = (denoised_snr[i] / original_snr[i] - 1) * 100
                self.stats_table.setItem(i, 4, QTableWidgetItem(f"{improvement_pct:+.1f}"))
            else:
                self.stats_table.setItem(i, 4, QTableWidgetItem("N/A"))
    
    def plot_snr_comparison(self):
        """绘制SNR对比图"""
        if not self.snr_results:
            return
            
        self.snr_figure.clear()
        
        wavelength = self.snr_results['wavelength']
        original_snr = self.snr_results['original_snr']
        denoised_snr = self.snr_results['denoised_snr']
        
        # 上左：SNR对比
        ax1 = self.snr_figure.add_subplot(221)
        ax1.plot(wavelength, original_snr, 'r-', alpha=0.7, linewidth=1, label='处理前')
        ax1.plot(wavelength, denoised_snr, 'b-', linewidth=1.5, label='处理后')
        ax1.set_xlabel('波长 (nm)')
        ax1.set_ylabel('信噪比 SNR')
        ax1.set_title('批量处理SNR对比')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 上右：SNR改善
        ax2 = self.snr_figure.add_subplot(222)
        improvement = self.snr_results['snr_improvement']
        ax2.plot(wavelength, improvement, 'g-', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('波长 (nm)')
        ax2.set_ylabel('SNR改善')
        ax2.set_title('SNR改善量')
        ax2.grid(True, alpha=0.3)
        
        # 下左：统计直方图
        ax3 = self.snr_figure.add_subplot(223)
        ax3.hist(original_snr, bins=30, alpha=0.7, label='处理前', color='red')
        ax3.hist(denoised_snr, bins=30, alpha=0.7, label='处理后', color='blue')
        ax3.set_xlabel('信噪比 SNR')
        ax3.set_ylabel('频次')
        ax3.set_title('SNR分布直方图')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 下右：累积分布
        ax4 = self.snr_figure.add_subplot(224)
        sorted_orig = np.sort(original_snr)
        sorted_denoised = np.sort(denoised_snr)
        y_vals = np.arange(1, len(sorted_orig) + 1) / len(sorted_orig)
        
        ax4.plot(sorted_orig, y_vals, 'r-', alpha=0.7, label='处理前')
        ax4.plot(sorted_denoised, y_vals, 'b-', linewidth=1.5, label='处理后')
        ax4.set_xlabel('信噪比 SNR')
        ax4.set_ylabel('累积概率')
        ax4.set_title('SNR累积分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        self.snr_figure.tight_layout()
        self.snr_canvas.draw()
    
    def plot_improvement(self):
        """绘制改善分析图"""
        if not self.snr_results:
            return
            
        self.snr_figure.clear()
        
        wavelength = self.snr_results['wavelength']
        improvement = self.snr_results['snr_improvement']
        original_snr = self.snr_results['original_snr']
        denoised_snr = self.snr_results['denoised_snr']
        
        # 改善百分比
        improvement_pct = np.divide(denoised_snr - original_snr, original_snr, 
                                  out=np.zeros_like(original_snr),
                                  where=original_snr > 1e-10) * 100
        
        # 上图：改善量和改善百分比
        ax1 = self.snr_figure.add_subplot(211)
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(wavelength, improvement, 'g-', linewidth=2, label='改善量')
        line2 = ax1_twin.plot(wavelength, improvement_pct, 'purple', alpha=0.7, label='改善百分比')
        
        ax1.set_xlabel('波长 (nm)', fontfamily='Microsoft YaHei')
        ax1.set_ylabel('SNR改善量', color='g', fontfamily='Microsoft YaHei')
        ax1_twin.set_ylabel('改善百分比 (%)', color='purple', fontfamily='Microsoft YaHei')
        ax1.set_title('SNR改善分析', fontfamily='Microsoft YaHei')
        ax1.grid(True, alpha=0.3)
        
        # 合并图例
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        # 下图：改善分布
        ax2 = self.snr_figure.add_subplot(212)
        ax2.hist(improvement, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('SNR改善量', fontfamily='Microsoft YaHei')
        ax2.set_ylabel('频次', fontfamily='Microsoft YaHei')
        ax2.set_title('改善量分布', fontfamily='Microsoft YaHei')
        ax2.grid(True, alpha=0.3)
        
        self.snr_figure.tight_layout()
        self.snr_canvas.draw()
    
    def plot_statistical_analysis(self):
        """绘制统计分析图"""
        if not self.snr_results:
            return
            
        self.snr_figure.clear()
        
        original_snr = self.snr_results['original_snr']
        denoised_snr = self.snr_results['denoised_snr']
        wavelength = self.snr_results['wavelength']
        
        # 过滤有效数据
        valid_mask = (np.isfinite(original_snr) & np.isfinite(denoised_snr) & 
                     (original_snr > 1e-10) & (denoised_snr > 1e-10))
        
        if np.sum(valid_mask) == 0:
            return
            
        orig_valid = original_snr[valid_mask]
        denoised_valid = denoised_snr[valid_mask]
        wavelength_valid = wavelength[valid_mask]
        
        # 散点图：处理前vs处理后
        ax1 = self.snr_figure.add_subplot(221)
        ax1.scatter(orig_valid, denoised_valid, alpha=0.6, s=20)
        min_val = min(orig_valid.min(), denoised_valid.min())
        max_val = max(orig_valid.max(), denoised_valid.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        ax1.set_xlabel('处理前SNR', fontfamily='Microsoft YaHei')
        ax1.set_ylabel('处理后SNR', fontfamily='Microsoft YaHei')
        ax1.set_title('SNR相关性分析', fontfamily='Microsoft YaHei')
        ax1.grid(True, alpha=0.3)
        
        # 箱线图
        ax2 = self.snr_figure.add_subplot(222)
        box_data = [orig_valid, denoised_valid]
        box_labels = ['处理前', '处理后']
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][1].set_facecolor('lightblue')
        ax2.set_ylabel('信噪比 SNR', fontfamily='Microsoft YaHei')
        ax2.set_title('SNR分布箱线图', fontfamily='Microsoft YaHei')
        ax2.grid(True, alpha=0.3)
        
        # 波长区域分析
        ax3 = self.snr_figure.add_subplot(223)
        # 将波长分为几个区域进行分析
        n_regions = 5
        region_size = len(wavelength_valid) // n_regions
        region_labels = []
        region_orig_means = []
        region_denoised_means = []
        
        for i in range(n_regions):
            start_idx = i * region_size
            end_idx = (i + 1) * region_size if i < n_regions - 1 else len(wavelength_valid)
            region_wavelengths = wavelength_valid[start_idx:end_idx]
            
            region_label = f"{region_wavelengths[0]:.0f}-{region_wavelengths[-1]:.0f}nm"
            region_labels.append(region_label)
            
            region_orig_means.append(np.mean(orig_valid[start_idx:end_idx]))
            region_denoised_means.append(np.mean(denoised_valid[start_idx:end_idx]))
        
        x_pos = np.arange(len(region_labels))
        width = 0.35
        ax3.bar(x_pos - width/2, region_orig_means, width, label='处理前', alpha=0.7)
        ax3.bar(x_pos + width/2, region_denoised_means, width, label='处理后', alpha=0.7)
        ax3.set_xlabel('波长区域', fontfamily='Microsoft YaHei')
        ax3.set_ylabel('平均SNR', fontfamily='Microsoft YaHei')
        ax3.set_title('不同波长区域SNR比较', fontfamily='Microsoft YaHei')
        ax3.set_xticklabels(region_labels, rotation=45, fontfamily='Microsoft YaHei')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 统计摘要
        ax4 = self.snr_figure.add_subplot(224)
        ax4.axis('off')
        
        stats_text = f"""统计摘要:
        
处理前SNR:
  平均值: {np.mean(orig_valid):.2f}
  标准差: {np.std(orig_valid):.2f}
  最小值: {np.min(orig_valid):.2f}
  最大值: {np.max(orig_valid):.2f}
  中位数: {np.median(orig_valid):.2f}

处理后SNR:
  平均值: {np.mean(denoised_valid):.2f}
  标准差: {np.std(denoised_valid):.2f}
  最小值: {np.min(denoised_valid):.2f}
  最大值: {np.max(denoised_valid):.2f}
  中位数: {np.median(denoised_valid):.2f}

总体改善:
  平均改善: {np.mean(denoised_valid - orig_valid):+.2f}
  改善比例: {((np.mean(denoised_valid) / np.mean(orig_valid)) - 1)*100:+.1f}%
"""
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='Microsoft YaHei')
        
        self.snr_figure.tight_layout()
        self.snr_canvas.draw()
    
    def plot_spectra_comparison(self):
        """绘制光谱对比图 - 根据复选框状态显示"""
        if not self.snr_results:
            return
            
        self.data_figure.clear()
        
        # 检查复选框状态
        show_original = self.show_original_cb.isChecked()
        show_denoised = self.show_denoised_cb.isChecked()
        
        # 如果都不选，则显示提示
        if not show_original and not show_denoised:
            ax = self.data_figure.add_subplot(111)
            ax.text(0.5, 0.5, '请至少选择一个显示选项', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=14, color='red')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.data_figure.tight_layout()
            self.data_canvas.draw()
            return
        
        # 显示前几条光谱的对比（最多5条）
        n_show = min(5, self.snr_results['original_mean'].shape[0] if len(self.snr_results['original_mean'].shape) > 1 else 1)
        
        ax1 = self.data_figure.add_subplot(211)
        
        # 根据复选框状态显示光谱
        if show_original:
            ax1.plot(self.snr_results['wavelength'], self.snr_results['original_mean'], 
                    'r-', alpha=0.7, linewidth=1.5, label='原始平均光谱')
            ax1.fill_between(self.snr_results['wavelength'], 
                            self.snr_results['original_mean'] - self.snr_results['original_std'],
                            self.snr_results['original_mean'] + self.snr_results['original_std'],
                            alpha=0.2, color='red', label='原始±标准差')
        
        if show_denoised:
            ax1.plot(self.snr_results['wavelength'], self.snr_results['denoised_mean'], 
                    'b-', linewidth=2, label='去噪平均光谱')
            ax1.fill_between(self.snr_results['wavelength'], 
                            self.snr_results['denoised_mean'] - self.snr_results['denoised_std'],
                            self.snr_results['denoised_mean'] + self.snr_results['denoised_std'],
                            alpha=0.2, color='blue', label='去噪±标准差')
        
        ax1.set_xlabel('波长 (nm)', fontfamily='Microsoft YaHei')
        ax1.set_ylabel('强度值', fontfamily='Microsoft YaHei')
        ax1.set_title('平均光谱对比', fontfamily='Microsoft YaHei')
        ax1.legend(prop={'family': 'Microsoft YaHei'})
        ax1.grid(True, alpha=0.3)
        
        # 标准差对比
        ax2 = self.data_figure.add_subplot(212)
        if show_original:
            ax2.plot(self.snr_results['wavelength'], self.snr_results['original_std'], 
                    'r-', alpha=0.7, linewidth=1.5, label='原始标准差')
        if show_denoised:
            ax2.plot(self.snr_results['wavelength'], self.snr_results['denoised_std'], 
                    'b-', linewidth=2, label='去噪标准差')
        ax2.set_xlabel('波长 (nm)', fontfamily='Microsoft YaHei')
        ax2.set_ylabel('标准差', fontfamily='Microsoft YaHei')
        ax2.set_title('标准差对比', fontfamily='Microsoft YaHei')
        ax2.legend(prop={'family': 'Microsoft YaHei'})
        ax2.grid(True, alpha=0.3)
        
        self.data_figure.tight_layout()
        self.data_canvas.draw()
    
    def plot_noise_analysis(self):
        """绘制噪声分析图"""
        if not self.snr_results:
            return
            
        self.data_figure.clear()
        
        # 噪声功率分析
        noise_power_orig = self.snr_results['original_std'] ** 2
        noise_power_denoised = self.snr_results['denoised_std'] ** 2
        noise_reduction = noise_power_orig - noise_power_denoised
        noise_reduction_pct = np.divide(noise_reduction, noise_power_orig, 
                                      out=np.zeros_like(noise_power_orig),
                                      where=noise_power_orig > 1e-10) * 100
        
        ax1 = self.data_figure.add_subplot(221)
        ax1.plot(self.snr_results['wavelength'], noise_power_orig, 'r-', alpha=0.7, label='原始噪声功率')
        ax1.plot(self.snr_results['wavelength'], noise_power_denoised, 'b-', linewidth=1.5, label='去噪噪声功率')
        ax1.set_xlabel('波长 (nm)', fontfamily='Microsoft YaHei')
        ax1.set_ylabel('噪声功率', fontfamily='Microsoft YaHei')
        ax1.set_title('噪声功率对比', fontfamily='Microsoft YaHei')
        ax1.legend(prop={'family': 'Microsoft YaHei'})
        ax1.grid(True, alpha=0.3)
        
        ax2 = self.data_figure.add_subplot(222)
        ax2.plot(self.snr_results['wavelength'], noise_reduction, 'g-', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('波长 (nm)', fontfamily='Microsoft YaHei')
        ax2.set_ylabel('噪声减少量', fontfamily='Microsoft YaHei')
        ax2.set_title('噪声减少量', fontfamily='Microsoft YaHei')
        ax2.grid(True, alpha=0.3)
        
        ax3 = self.data_figure.add_subplot(223)
        ax3.plot(self.snr_results['wavelength'], noise_reduction_pct, 'purple', linewidth=1.5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('波长 (nm)', fontfamily='Microsoft YaHei')
        ax3.set_ylabel('噪声减少百分比 (%)', fontfamily='Microsoft YaHei')
        ax3.set_title('噪声减少百分比', fontfamily='Microsoft YaHei')
        ax3.grid(True, alpha=0.3)
        
        # 噪声分布直方图
        ax4 = self.data_figure.add_subplot(224)
        ax4.hist(noise_power_orig, bins=30, alpha=0.7, label='原始噪声', color='red')
        ax4.hist(noise_power_denoised, bins=30, alpha=0.7, label='去噪噪声', color='blue')
        ax4.set_xlabel('噪声功率', fontfamily='Microsoft YaHei')
        ax4.set_ylabel('频次', fontfamily='Microsoft YaHei')
        ax4.set_title('噪声功率分布', fontfamily='Microsoft YaHei')
        ax4.legend(prop={'family': 'Microsoft YaHei'})
        ax4.grid(True, alpha=0.3)
        
        self.data_figure.tight_layout()
        self.data_canvas.draw()
    
    def save_results(self):
        """保存结果"""
        if not self.snr_results:
            QMessageBox.warning(self, "警告", "没有处理结果可保存！")
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "batch_wavelet_snr_results.xlsx",
            "Excel文件 (*.xlsx);;所有文件 (*)"
        )
        
        if save_path:
            try:
                with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                    # SNR统计表
                    df_snr = pd.DataFrame({
                        '波长(nm)': self.snr_results['wavelength'],
                        '处理前SNR': self.snr_results['original_snr'],
                        '处理后SNR': self.snr_results['denoised_snr'],
                        'SNR改善量': self.snr_results['snr_improvement'],
                        '原始均值': self.snr_results['original_mean'],
                        '原始标准差': self.snr_results['original_std'],
                        '去噪均值': self.snr_results['denoised_mean'],
                        '去噪标准差': self.snr_results['denoised_std']
                    })
                    df_snr.to_excel(writer, sheet_name='SNR统计', index=False)
                    
                    # 统计摘要
                    valid_mask = (np.isfinite(self.snr_results['original_snr']) & 
                                np.isfinite(self.snr_results['denoised_snr']))
                    if np.sum(valid_mask) > 0:
                        orig_valid = self.snr_results['original_snr'][valid_mask]
                        denoised_valid = self.snr_results['denoised_snr'][valid_mask]
                        
                        summary_data = {
                            '统计项': ['平均值', '标准差', '最小值', '最大值', '中位数'],
                            '处理前SNR': [
                                np.mean(orig_valid),
                                np.std(orig_valid),
                                np.min(orig_valid),
                                np.max(orig_valid),
                                np.median(orig_valid)
                            ],
                            '处理后SNR': [
                                np.mean(denoised_valid),
                                np.std(denoised_valid),
                                np.min(denoised_valid),
                                np.max(denoised_valid),
                                np.median(denoised_valid)
                            ]
                        }
                        df_summary = pd.DataFrame(summary_data)
                        df_summary.to_excel(writer, sheet_name='统计摘要', index=False)
                
                QMessageBox.information(self, "成功", f"结果已保存到: {save_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

def main():
    """主函数"""
    app = QApplication([])
    app.setStyle('Fusion')
    
    # 设置字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    # 创建并显示主窗口
    window = CorrectedBatchWaveletGUI()
    window.show()
    
    # 运行应用
    app.exec_()

if __name__ == '__main__':
    main()