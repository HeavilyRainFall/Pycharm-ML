"""
光谱小波变换去噪 - 简化测试版本
用于快速验证算法正确性和效果
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

try:
    import pywt
except ImportError:
    print("警告: 未安装pywavelets库，将使用scipy替代")
    from scipy import signal

# ----------------------核心算法实现----------------------
def generate_test_spectrum(length=1024, noise_level=0.1):
    """生成测试光谱数据（含噪声）"""
    # 生成模拟光谱信号
    x = np.linspace(400, 1000, length)  # 波长范围400-1000nm
    # 模拟多个峰的叠加
    spectrum = (np.sin(0.02 * x) * np.exp(-(x-600)**2/2000) + 
                0.8 * np.sin(0.015 * x) * np.exp(-(x-750)**2/1500) +
                0.6 * np.sin(0.01 * x) * np.exp(-(x-500)**2/3000))
    
    # 添加噪声
    noise = np.random.normal(0, noise_level, length)
    noisy_spectrum = spectrum + noise
    
    return x, spectrum, noisy_spectrum

def soft_threshold_py(coef, threshold):
    """Python版本软阈值函数"""
    if abs(coef) <= threshold:
        return 0.0
    return np.sign(coef) * (abs(coef) - threshold)

def spectral_wavelet_denoise_py(spectrum, wavelet='db4', level=6):
    """
    Python版本光谱小波去噪（简化实现）
    复现C语言核心逻辑
    """
    try:
        # 小波分解
        coeffs = pywt.wavedec(spectrum, wavelet, level=level, mode='symmetric')
        
        # 获取最后一层细节系数用于阈值计算
        last_level_coeffs = coeffs[-1]
        
        # 自适应阈值计算（复现C语言逻辑）
        # 将系数分成10组计算标准差
        coef_len = len(last_level_coeffs)
        group_size = max(1, coef_len // 10)
        
        group_stds = []
        for i in range(0, coef_len, group_size):
            group = last_level_coeffs[i:i+group_size]
            if len(group) > 1:
                std_group = np.std(group, ddof=1)
                if std_group > 0:
                    group_stds.append(std_group)
        
        if len(group_stds) > 0:
            std_avg = np.mean(group_stds)
            std_of_stds = np.std(group_stds, ddof=1)
            if std_of_stds > 0:
                # 阈值公式：threshold = min(1000, (1.3 * std_avg/std_of_stds)^10)
                ratio = 1.3 * (std_avg / std_of_stds)
                threshold = min(1000.0, ratio ** 10)
            else:
                threshold = 0.1
        else:
            threshold = 0.1
        
        print(f"计算阈值: {threshold:.6f}")
        
        # 对细节系数应用软阈值
        coeffs_denoised = [coeffs[0]]  # 保留近似系数
        for detail_coeffs in coeffs[1:]:
            denoised_detail = np.array([soft_threshold_py(c, threshold) for c in detail_coeffs])
            coeffs_denoised.append(denoised_detail)
        
        # 小波重构
        denoised_spectrum = pywt.waverec(coeffs_denoised, wavelet, mode='symmetric')
        
        # 确保长度一致
        if len(denoised_spectrum) > len(spectrum):
            denoised_spectrum = denoised_spectrum[:len(spectrum)]
        elif len(denoised_spectrum) < len(spectrum):
            denoised_spectrum = np.pad(denoised_spectrum, 
                                     (0, len(spectrum) - len(denoised_spectrum)), 
                                     mode='edge')
        
        return denoised_spectrum, threshold
        
    except Exception as e:
        print(f"小波去噪失败: {e}")
        return spectrum.copy(), 0.0

def compare_methods_demo():
    """对比不同去噪方法的效果"""
    print("生成测试光谱数据...")
    wavelength, clean_spectrum, noisy_spectrum = generate_test_spectrum(length=512, noise_level=0.2)
    
    print("执行小波去噪...")
    denoised_spectrum, threshold = spectral_wavelet_denoise_py(noisy_spectrum)
    
    # 计算评价指标
    def calculate_rmse(original, processed):
        return np.sqrt(np.mean((original - processed) ** 2))
    
    def calculate_snr(signal, noise):
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        return 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    rmse_noisy = calculate_rmse(clean_spectrum, noisy_spectrum)
    rmse_denoised = calculate_rmse(clean_spectrum, denoised_spectrum)
    
    snr_noisy = calculate_snr(clean_spectrum, noisy_spectrum - clean_spectrum)
    snr_denoised = calculate_snr(clean_spectrum, denoised_spectrum - clean_spectrum)
    
    print(f"\n去噪效果评估:")
    print(f"噪声水平 RMSE: {rmse_noisy:.4f}")
    print(f"去噪后 RMSE: {rmse_denoised:.4f}")
    print(f"改善幅度: {((rmse_noisy - rmse_denoised) / rmse_noisy * 100):.1f}%")
    print(f"噪声SNR: {snr_noisy:.2f} dB")
    print(f"去噪后SNR: {snr_denoised:.2f} dB")
    print(f"SNR提升: {(snr_denoised - snr_noisy):.2f} dB")
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 原始信号对比
    plt.subplot(2, 2, 1)
    plt.plot(wavelength, clean_spectrum, 'g-', linewidth=2, label='真实信号')
    plt.plot(wavelength, noisy_spectrum, 'r-', alpha=0.7, linewidth=1, label='含噪信号')
    plt.xlabel('波长 (nm)')
    plt.ylabel('强度')
    plt.title('原始信号 vs 含噪信号')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 去噪效果
    plt.subplot(2, 2, 2)
    plt.plot(wavelength, clean_spectrum, 'g-', linewidth=2, label='真实信号')
    plt.plot(wavelength, denoised_spectrum, 'b-', linewidth=1.5, label='去噪信号')
    plt.xlabel('波长 (nm)')
    plt.ylabel('强度')
    plt.title(f'小波去噪效果 (阈值={threshold:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 误差分析
    plt.subplot(2, 2, 3)
    noise_error = noisy_spectrum - clean_spectrum
    denoise_error = denoised_spectrum - clean_spectrum
    plt.plot(wavelength, noise_error, 'r-', alpha=0.7, label='噪声误差')
    plt.plot(wavelength, denoise_error, 'b-', alpha=0.7, label='去噪误差')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('波长 (nm)')
    plt.ylabel('误差')
    plt.title('误差对比分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 频谱分析
    plt.subplot(2, 2, 4)
    # 计算频谱
    freq_orig = np.abs(np.fft.fft(clean_spectrum - np.mean(clean_spectrum)))
    freq_noisy = np.abs(np.fft.fft(noisy_spectrum - np.mean(noisy_spectrum)))
    freq_denoised = np.abs(np.fft.fft(denoised_spectrum - np.mean(denoised_spectrum)))
    
    freq_axis = np.fft.fftfreq(len(wavelength))
    plt.semilogy(freq_axis[:len(freq_axis)//2], 
                 freq_orig[:len(freq_orig)//2], 'g-', label='真实信号')
    plt.semilogy(freq_axis[:len(freq_axis)//2], 
                 freq_noisy[:len(freq_noisy)//2], 'r-', alpha=0.7, label='含噪信号')
    plt.semilogy(freq_axis[:len(freq_axis)//2], 
                 freq_denoised[:len(freq_denoised)//2], 'b-', alpha=0.7, label='去噪信号')
    plt.xlabel('频率')
    plt.ylabel('幅值 (log scale)')
    plt.title('频谱对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('wavelet_denoise_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'wavelength': wavelength,
        'clean': clean_spectrum,
        'noisy': noisy_spectrum,
        'denoised': denoised_spectrum,
        'metrics': {
            'rmse_noisy': rmse_noisy,
            'rmse_denoised': rmse_denoised,
            'snr_noisy': snr_noisy,
            'snr_denoised': snr_denoised,
            'threshold': threshold
        }
    }

# ----------------------简单使用示例----------------------
def simple_example():
    """简单使用示例"""
    print("=== 光谱小波去噪简单示例 ===")
    
    # 生成示例数据
    x = np.linspace(400, 800, 256)
    # 模拟光谱：几个高斯峰叠加
    spectrum = (np.exp(-(x-500)**2/50) + 
                0.8*np.exp(-(x-600)**2/80) + 
                0.6*np.exp(-(x-700)**2/60))
    
    # 添加噪声
    noisy_spectrum = spectrum + np.random.normal(0, 0.1, len(spectrum))
    
    # 去噪
    denoised, threshold = spectral_wavelet_denoise_py(noisy_spectrum)
    
    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(x, spectrum, 'g-', linewidth=2, label='原始光谱')
    plt.plot(x, noisy_spectrum, 'r-', alpha=0.7, label='含噪光谱')
    plt.plot(x, denoised, 'b-', linewidth=1.5, label=f'去噪光谱 (阈值={threshold:.3f})')
    plt.xlabel('波长 (nm)')
    plt.ylabel('强度')
    plt.title('光谱小波去噪示例')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"使用的阈值: {threshold:.6f}")

if __name__ == "__main__":
    print("选择运行模式:")
    print("1. 简单示例")
    print("2. 详细对比分析")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        simple_example()
    elif choice == "2":
        results = compare_methods_demo()
    else:
        print("无效选择，运行简单示例")
        simple_example()
