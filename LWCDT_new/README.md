# 光谱小波变换去噪程序使用说明

## 📋 程序概述

本程序实现了基于C语言小波变换算法的光谱数据去噪处理，完全复现了您提供的C代码逻辑，包括自适应阈值计算和软阈值去噪等核心技术。

## 📁 文件结构

```
LWCDT_new/
├── wavemin/                    # C语言小波变换源码
│   ├── waveaux.c              # 辅助函数实现
│   ├── waveaux.h              # 辅助函数声明
│   ├── wavemin.c              # 核心小波变换实现
│   └── wavemin.h              # 核心函数声明
├── test.c                     # C语言调用示例
├── spectral_wavelet_denoise.py # 完整版Python实现
├── simple_wavelet_demo.py     # 简化测试版本
├── requirements.txt           # Python依赖包
└── README.md                  # 本说明文档
```

## 🔧 核心算法原理

### 1. 小波分解过程
- 使用db4小波基函数
- 6层分解（D1-D6细节系数 + A近似系数）
- 对称延拓边界处理

### 2. 自适应阈值计算（核心创新）
```
算法步骤：
1. 取最后一层细节系数D6
2. 将D6分成10个等份组
3. 计算每组标准差：σ₁, σ₂, ..., σ₁₀
4. 计算统计特征：
   - 平均标准差：avg_std = mean(σ₁...σ₁₀)
   - 标准差的标准差：std_of_stds = std(σ₁...σ₁₀)
5. 阈值公式：threshold = min(1000, (1.3 × avg_std/std_of_stds)¹⁰)
```

### 3. 软阈值去噪
```
处理函数：
thresh(x, λ) = sign(x) × max(|x| - λ, 0)
其中λ为计算得到的自适应阈值
```

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据文件
支持CSV和Excel格式，数据格式示例：

**CSV格式 (spectra.csv):**
```csv
wavelength,spectrum1,spectrum2,spectrum3
400.0,0.123,0.456,0.789
401.0,0.125,0.458,0.791
...
```

**Excel格式 (spectra.xlsx):**
- 第一列为波长数据
- 后续列为不同光谱数据

### 3. 运行简化示例
```bash
python simple_wavelet_demo.py
```

### 4. 运行完整程序
```python
from spectral_wavelet_denoise import main

# 配置参数
results = main(
    file_path="your_spectra_data.csv",
    wavelength_col="wavelength",
    spectrum_cols=None,  # None表示自动识别所有光谱列
    wavelet='db4',
    level=6,
    save_results=True
)
```

## 📊 输出结果

程序会生成以下输出：

1. **控制台信息**：
   - 处理进度
   - 计算得到的自适应阈值
   - 信噪比统计

2. **可视化图表**：
   - 去噪前后光谱对比图
   - 误差分析图
   - 信噪比分布图

3. **数据文件**：
   - 去噪后的CSV/Excel文件
   - 包含原始数据和去噪数据

## 🛠️ 高级使用

### 自定义参数调整

```python
from spectral_wavelet_denoise import SpectralWaveletDenoiser

# 创建去噪器实例
denoiser = SpectralWaveletDenoiser(
    wavelet='db4',    # 小波基函数 ('db4', 'haar', 'db2'等)
    level=6,          # 分解层数 (1-10)
    extension='sym'   # 延拓方式 ('sym', 'per')
)

# 处理单条光谱
denoised_spectrum = denoiser.denoise_single_spectrum(your_spectrum)

# 批量处理
denoised_spectra = denoiser.batch_denoise(spectra_matrix)
```

### 不同小波基函数选择

| 小波类型 | 特点 | 适用场景 |
|---------|------|----------|
| db4 | 平滑性好，支撑长度适中 | 一般光谱去噪（推荐） |
| haar | 计算最快，但平滑性差 | 快速预处理 |
| db2/db3 | 支撑长度短 | 高频噪声较多 |
| db5-db10 | 平滑性更好 | 需要精细处理 |

### 分解层数选择

```python
# 根据数据长度自动选择
import pywt
max_level = pywt.dwt_max_level(data_length, 'db4')

# 推荐设置：
# 数据点数 < 128:  level=4
# 数据点数 128-512: level=5  
# 数据点数 512-2048: level=6
# 数据点数 > 2048:  level=7-8
```

## 📈 性能优化建议

### 1. 数据预处理
```python
# 去除异常值
from scipy import stats
def remove_outliers(spectrum, threshold=3):
    z_scores = np.abs(stats.zscore(spectrum))
    return spectrum[z_scores < threshold]

# 基线校正
def baseline_correction(spectrum):
    # 简单线性基线校正
    x = np.arange(len(spectrum))
    coeffs = np.polyfit(x, spectrum, 1)
    baseline = np.polyval(coeffs, x)
    return spectrum - baseline
```

### 2. 批量处理优化
```python
# 使用并行处理加速大批量数据
from multiprocessing import Pool

def process_spectrum_batch(spectra_list):
    denoiser = SpectralWaveletDenoiser()
    with Pool() as pool:
        results = pool.map(denoiser.denoise_single_spectrum, spectra_list)
    return np.array(results)
```

## 🐛 常见问题解决

### 1. 导入库失败
```
错误: ModuleNotFoundError: No module named 'pywt'
解决: pip install PyWavelets
```

### 2. 数据格式错误
```
错误: KeyError: 'wavelength'
解决: 检查数据文件中是否存在指定的波长列名
```

### 3. 内存不足
```
错误: MemoryError
解决: 
- 减少批量处理的数据量
- 降低分解层数
- 使用更小的数据块逐段处理
```

### 4. 去噪效果不佳
```
可能原因及解决方案:
1. 阈值过高 → 降低level参数或尝试其他小波基
2. 噪声类型特殊 → 调整预处理步骤
3. 信号太弱 → 检查数据质量和采集条件
```

## 🔍 算法验证

### 对比实验建议

```python
# 与其他去噪方法对比
methods = {
    'wavelet': lambda x: spectral_wavelet_denoise_py(x)[0],
    'savitzky_golay': lambda x: signal.savgol_filter(x, 11, 3),
    'median': lambda x: signal.medfilt(x, 5),
    'gaussian': lambda x: gaussian_filter1d(x, sigma=2)
}

# 评估指标
def evaluate_denoising(original, noisy, denoised):
    rmse = np.sqrt(np.mean((original - denoised)**2))
    snr = 10 * np.log10(np.var(original) / np.var(denoised - original))
    correlation = np.corrcoef(original, denoised)[0,1]
    return {'RMSE': rmse, 'SNR': snr, 'Correlation': correlation}
```

## 📞 技术支持

如有问题，请提供：
1. 错误信息截图
2. 数据文件格式示例
3. 使用的参数配置
4. 期望的处理效果描述

## 📄 许可证

本程序基于MIT许可证开源，可自由使用和修改。

---
*程序版本: 1.0*
*最后更新: 2024年*
