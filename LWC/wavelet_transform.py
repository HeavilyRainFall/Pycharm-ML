"""
小波变换核心模块
移植自C语言的wavemin库
实现DB4小波变换和逆变换功能
"""

import numpy as np
from scipy import signal
import math

class WaveletFilter:
    """小波滤波器类"""
    
    # DB4小波系数
    DB4_LPD = [
        0.2303778133088965, 0.7148465705529156, 0.6308807679298589, 
        -0.02798376941685985, -0.1870348117190931, 0.03084138183556076,
        0.03288301166688520, -0.01059740178506903
    ]
    
    DB4_HPD = [
        -0.01059740178506903, -0.03288301166688520, 0.03084138183556076,
        0.1870348117190931, -0.02798376941685985, -0.6308807679298589,
        0.7148465705529156, -0.2303778133088965
    ]
    
    DB4_LPR = [
        0.2303778133088965, 0.7148465705529156, 0.6308807679298589,
        -0.02798376941685985, -0.1870348117190931, 0.03084138183556076,
        0.03288301166688520, -0.01059740178506903
    ]
    
    DB4_HPR = [
        0.01059740178506903, -0.03288301166688520, -0.03084138183556076,
        0.1870348117190931, 0.02798376941685985, -0.6308807679298589,
        -0.7148465705529156, -0.2303778133088965
    ]
    
    def __init__(self, wavelet_name='db4'):
        """初始化小波滤波器"""
        self.wavelet_name = wavelet_name
        if wavelet_name.lower() == 'db4':
            self.lpd = np.array(self.DB4_LPD)
            self.hpd = np.array(self.DB4_HPD)
            self.lpr = np.array(self.DB4_LPR)
            self.hpr = np.array(self.DB4_HPR)
            self.filter_length = len(self.lpd)
        else:
            raise ValueError(f"不支持的小波类型: {wavelet_name}")

class WaveletTransform:
    """小波变换类"""
    
    def __init__(self, wavelet_name='db4'):
        """初始化小波变换对象"""
        self.wavelet = WaveletFilter(wavelet_name)
        self.coefficients = None
        self.lengths = None
        self.levels = 0
        
    def dwt(self, signal_data, levels=6, extension='sym'):
        """
        离散小波变换
        
        参数:
        signal_data: 输入信号数组
        levels: 分解层数
        extension: 延拓方式 ('sym' 或 'per')
        
        返回:
        coefficients: 小波系数数组
        lengths: 各层长度信息
        """
        sig = np.array(signal_data, dtype=float)
        N = len(sig)
        
        # 计算最大迭代次数
        max_iter = self._wmaxiter(N, self.wavelet.filter_length)
        if levels > max_iter:
            levels = max_iter
            
        self.levels = levels
        
        # 初始化输出数组
        output_size = N + 2 * levels * (self.wavelet.filter_length + 1)
        output = np.zeros(output_size)
        
        # 存储各层长度
        self.lengths = [0] * (levels + 2)
        self.lengths[levels + 1] = N
        
        temp_sig = sig.copy()
        temp_len = N
        out_index = output_size
        
        if extension == 'per':
            # 周期延拓
            for level in range(levels):
                new_len = math.ceil(temp_len / 2.0)
                self.lengths[levels - level] = new_len
                out_index -= new_len
                
                # 执行一层DWT
                cA, cD = self._dwt_per_level(temp_sig[:temp_len], new_len)
                
                output[out_index:out_index + new_len] = cD
                temp_sig[:new_len] = cA
                temp_len = new_len
                
            # 最后一层近似系数
            self.lengths[0] = self.lengths[1]
            output[:self.lengths[0]] = temp_sig[:self.lengths[0]]
            
        elif extension == 'sym':
            # 对称延拓
            lp = self.wavelet.filter_length
            for level in range(levels):
                new_len = math.ceil((temp_len + lp - 2) / 2.0)
                self.lengths[levels - level] = new_len
                out_index -= new_len
                
                # 执行一层DWT
                cA, cD = self._dwt_sym_level(temp_sig[:temp_len], new_len)
                
                output[out_index:out_index + new_len] = cD
                temp_sig[:new_len] = cA
                temp_len = new_len
                
            # 最后一层近似系数
            self.lengths[0] = self.lengths[1]
            output[:self.lengths[0]] = temp_sig[:self.lengths[0]]
            
        self.coefficients = output
        return output, self.lengths
    
    def idwt(self, coefficients=None, lengths=None, extension='sym', preserve_length=True):
        """
        逆离散小波变换
        
        参数:
        coefficients: 小波系数数组（如果不提供则使用上次变换的结果）
        lengths: 各层长度信息
        extension: 延拓方式
        preserve_length: 是否保持原始长度（默认True）
        
        返回:
        重构信号
        """
        if coefficients is None:
            coefficients = self.coefficients
        if lengths is None:
            lengths = self.lengths
            
        if coefficients is None or lengths is None:
            raise ValueError("需要提供小波系数和长度信息")
            
        original_length = lengths[-1]  # 原始信号长度
        J = self.levels
        app_len = lengths[0]
        det_len = lengths[1]
        
        # 初始化重构信号
        reconstructed = np.zeros(lengths[J + 1])
        reconstructed[:app_len] = coefficients[:app_len]
        
        iter_pos = app_len
        
        if extension == 'per':
            # 周期延拓逆变换
            for i in range(J):
                cA = reconstructed[:det_len]
                cD = coefficients[iter_pos:iter_pos + det_len]
                
                # 执行一层IDWT
                reconstructed = self._idwt_per_level(cA, cD, det_len)
                
                iter_pos += det_len
                det_len = lengths[i + 2]
                
        elif extension == 'sym':
            # 对称延拓逆变换
            for i in range(J):
                cA = reconstructed[:det_len]
                cD = coefficients[iter_pos:iter_pos + det_len]
                
                # 执行一层IDWT
                reconstructed = self._idwt_sym_level(cA, cD, det_len, i)
                
                iter_pos += det_len
                det_len = lengths[i + 2]
        
        # 如果需要保持原始长度，截取相应部分
        if preserve_length and len(reconstructed) != original_length:
            # 通常多余的部分在末尾，取前面的部分
            reconstructed = reconstructed[:original_length]
            
        return reconstructed
    
    def _dwt_per_level(self, sig, output_len):
        """周期延拓方式的一层DWT"""
        N = len(sig)
        lpd = self.wavelet.lpd
        hpd = self.wavelet.hpd
        flen = len(lpd)
        l2 = flen // 2
        
        cA = np.zeros(output_len)
        cD = np.zeros(output_len)
        
        is_odd = N % 2
        
        for i in range(output_len):
            t = 2 * i + l2
            for l in range(flen):
                if (t - l) >= l2 and (t - l) < N:
                    cA[i] += lpd[l] * sig[t - l]
                    cD[i] += hpd[l] * sig[t - l]
                elif (t - l) < l2 and (t - l) >= 0:
                    cA[i] += lpd[l] * sig[t - l]
                    cD[i] += hpd[l] * sig[t - l]
                elif (t - l) < 0 and is_odd == 0:
                    cA[i] += lpd[l] * sig[t - l + N]
                    cD[i] += hpd[l] * sig[t - l + N]
                elif (t - l) < 0 and is_odd == 1:
                    if (t - l) != -1:
                        cA[i] += lpd[l] * sig[t - l + N + 1]
                        cD[i] += hpd[l] * sig[t - l + N + 1]
                    else:
                        cA[i] += lpd[l] * sig[N - 1]
                        cD[i] += hpd[l] * sig[N - 1]
                elif (t - l) >= N and is_odd == 0:
                    cA[i] += lpd[l] * sig[t - l - N]
                    cD[i] += hpd[l] * sig[t - l - N]
                elif (t - l) >= N and is_odd == 1:
                    if t - l != N:
                        cA[i] += lpd[l] * sig[t - l - (N + 1)]
                        cD[i] += hpd[l] * sig[t - l - (N + 1)]
                    else:
                        cA[i] += lpd[l] * sig[N - 1]
                        cD[i] += hpd[l] * sig[N - 1]
                        
        return cA, cD
    
    def _dwt_sym_level(self, sig, output_len):
        """对称延拓方式的一层DWT"""
        N = len(sig)
        lpd = self.wavelet.lpd
        hpd = self.wavelet.hpd
        flen = len(lpd)
        
        cA = np.zeros(output_len)
        cD = np.zeros(output_len)
        
        for i in range(output_len):
            t = 2 * i + 1
            for l in range(flen):
                if (t - l) >= 0 and (t - l) < N:
                    cA[i] += lpd[l] * sig[t - l]
                    cD[i] += hpd[l] * sig[t - l]
                elif (t - l) < 0:
                    cA[i] += lpd[l] * sig[-t + l - 1]
                    cD[i] += hpd[l] * sig[-t + l - 1]
                elif (t - l) >= N:
                    cA[i] += lpd[l] * sig[2 * N - t + l - 1]
                    cD[i] += hpd[l] * sig[2 * N - t + l - 1]
                    
        return cA, cD
    
    def _idwt_per_level(self, cA, cD, length):
        """周期延拓方式的一层IDWT"""
        flen = len(self.wavelet.lpr)
        l2 = flen // 2
        output_len = 2 * length
        result = np.zeros(output_len + 2 * l2 - 1)
        
        m = -2
        n = -1
        
        for i in range(length + l2 - 1):
            m += 2
            n += 2
            for l in range(l2):
                t = 2 * l
                idx = i - l
                if idx >= 0 and idx < length:
                    result[m] += (self.wavelet.lpr[t] * cA[idx] + 
                                self.wavelet.hpr[t] * cD[idx])
                    result[n] += (self.wavelet.lpr[t + 1] * cA[idx] + 
                                self.wavelet.hpr[t + 1] * cD[idx])
                elif idx >= length and idx < length + flen - 1:
                    result[m] += (self.wavelet.lpr[t] * cA[idx - length] + 
                                self.wavelet.hpr[t] * cD[idx - length])
                    result[n] += (self.wavelet.lpr[t + 1] * cA[idx - length] + 
                                self.wavelet.hpr[t + 1] * cD[idx - length])
                elif idx < 0 and idx > -l2:
                    result[m] += (self.wavelet.lpr[t] * cA[length + idx] + 
                                self.wavelet.hpr[t] * cD[length + idx])
                    result[n] += (self.wavelet.lpr[t + 1] * cA[length + idx] + 
                                self.wavelet.hpr[t + 1] * cD[length + idx])
                    
        # 提取有效部分
        final_result = np.zeros(2 * length)
        for k in range(l2 - 1, 2 * length + l2 - 1):
            final_result[k - l2 + 1] = result[k]
            
        return final_result
    
    def _idwt_sym_level(self, cA, cD, length, level):
        """对称延拓方式的一层IDWT"""
        flen = len(self.wavelet.lpr)
        output_len = 2 * length - 1
        result = np.zeros(output_len + 2 * flen - 1)
        
        m = -2
        n = -1
        
        for v in range(length):
            i = v
            m += 2
            n += 2
            for l in range(flen // 2):
                t = 2 * l
                idx = i - l
                if idx >= 0 and idx < length:
                    # 应用阈值处理（软阈值）
                    # 注意：cD[idx]是单个系数，不能直接用于分组计算
                    # 对于单个系数，使用简化阈值计算
                    if len(cD) > 100:  # 只有当细节系数足够长时才使用分组算法
                        threshold = self._calculate_threshold(cD, level)
                    else:
                        # 单个系数或短序列，使用固定阈值
                        threshold = 1.0
                    processed_cD = self._soft_threshold(cD[idx], threshold)
                    
                    result[m] += (self.wavelet.lpr[t] * cA[idx] + 
                                self.wavelet.hpr[t] * processed_cD)
                    result[n] += (self.wavelet.lpr[t + 1] * cA[idx] + 
                                self.wavelet.hpr[t + 1] * processed_cD)
                    
        # 提取有效部分
        final_result = np.zeros(2 * length - 1)
        lf = flen
        for k in range(lf - 2, 2 * length):
            final_result[k - lf + 2] = result[k]
            
        return final_result
    
    def _calculate_threshold(self, coeffs, level, group_size=100):
        """计算阈值（根据您提供的小波自动阈值算法）"""
        # 1. 将细节系数按每group_size个像素分组
        groups = []
        for i in range(0, len(coeffs), group_size):
            group = coeffs[i:i+group_size]
            groups.append(group)
        
        # 2. 对每组求std和平均值
        stds = []
        means = []
        for group in groups:
            if len(group) > 0:
                std_val = np.std(group)
                mean_val = np.mean(group)
                stds.append(std_val)
                means.append(mean_val)
        
        # 3. 计算阈值 t = (1.3 * s̄ / σₛ)^10
        if len(means) == 0:
            return 1000
            
        s_bar = np.mean(means)  # 平均值的平均
        sigma_s = np.std(means)  # 平均值的标准差
        
        if sigma_s == 0:
            t = 1000  # 避免除零
        else:
            t = (1.3 * s_bar / sigma_s) ** 10
            
        # 4. 如果t > 1000，则t = 1000
        if t > 1000:
            t = 1000
            
        return t
    
    def _soft_threshold(self, coef, thr):
        """软阈值函数"""
        if abs(coef) <= thr:
            return 0.0
        elif coef > 0:
            return coef - thr
        else:
            return coef + thr
    
    def _wmaxiter(self, sig_len, filt_len):
        """计算最大迭代次数"""
        temp = math.log(sig_len / (filt_len - 1.0)) / math.log(2.0)
        return int(temp)

def calculate_snr(signal_data, use_ratio=True):
    """
    计算信噪比
    
    参数:
    signal_data: 信号数据
    use_ratio: 是否使用比例形式（True）还是dB形式（False）
    
    返回:
    SNR: 信噪比（比例或dB）
    """
    signal_data = np.array(signal_data, dtype=float)
    
    # 检查数据长度
    if len(signal_data) < 2:
        return 0.0  # 单帧数据无法计算信噪比
    
    # 计算信号均值
    mean_signal = np.mean(signal_data)
    
    # 计算信号功率
    signal_power = np.mean((signal_data - mean_signal) ** 2)
    
    # 计算噪声功率（使用相邻点差分估算噪声）
    noise = np.diff(signal_data)
    noise_power = np.var(noise)
    
    # 避免除零错误
    if noise_power == 0 or signal_power == 0:
        return 0.0
    
    # 计算SNR
    if use_ratio:
        snr = signal_power / noise_power
    else:
        snr = 10 * np.log10(signal_power / noise_power)
    
    return snr

def apply_wavelet_denoising(data, levels=6, wavelet_name='db4'):
    """
    应用小波去噪
    
    参数:
    data: 输入数据
    levels: 分解层数
    wavelet_name: 小波类型
    
    返回:
    denoised_data: 去噪后的数据
    """
    # 创建小波变换对象
    wt = WaveletTransform(wavelet_name)
    
    # 执行小波变换
    coeffs, lengths = wt.dwt(data, levels=levels, extension='sym')
    
    # 执行逆变换（包含阈值处理）
    denoised_data = wt.idwt(coeffs, lengths, extension='sym')
    
    return denoised_data

def calculate_wavelength_snr_spectrum(original_data, denoised_data, window_size=50):
    """
    计算每个波长点的局部信噪比谱
    
    参数:
    original_data: 原始光谱数据 (samples × wavelengths)
    denoised_data: 去噪后的光谱数据 (samples × wavelengths)
    window_size: 滑动窗口大小，用于局部统计
    
    返回:
    wavelength_snr: 每个波长点的信噪比 (dB)
    wavelength_values: 波长值数组
    """
    if len(original_data.shape) != 2:
        raise ValueError("输入数据必须是二维数组 (samples × wavelengths)")
    
    samples, wavelengths = original_data.shape
    
    # 如果波长维度小于窗口大小，调整窗口
    if wavelengths < window_size:
        window_size = max(5, wavelengths // 10)
    
    # 初始化输出数组
    wavelength_snr = np.zeros(wavelengths)
    
    # 对每个波长点计算局部信噪比
    for i in range(wavelengths):
        # 确定局部窗口范围
        start_idx = max(0, i - window_size // 2)
        end_idx = min(wavelengths, i + window_size // 2 + 1)
        
        # 提取局部区域的数据
        local_original = original_data[:, start_idx:end_idx]
        local_denoised = denoised_data[:, start_idx:end_idx]
        
        # 计算局部噪声和信号统计
        local_noise = local_original - local_denoised
        
        # 局部噪声功率（方差）
        noise_var = np.var(local_noise, axis=0)
        mean_noise_var = np.mean(noise_var)
        
        # 局部信号功率（方差）
        signal_var = np.var(local_denoised, axis=0)
        mean_signal_var = np.mean(signal_var)
        
        # 计算该波长点的信噪比（比值形式）
        if mean_noise_var > 0:
            snr_linear = mean_signal_var / mean_noise_var
            wavelength_snr[i] = snr_linear
        else:
            wavelength_snr[i] = 1e6  # 设置一个较大的默认值避免无穷大
    
    # 生成波长轴（假设均匀分布）
    wavelength_values = np.linspace(400, 1000, wavelengths)  # 假设400-1000nm范围
    
    return wavelength_snr, wavelength_values

def calculate_pointwise_snr(original_data, denoised_data):
    """
    计算每个波长点的逐点信噪比（基于样本间的变异性）
    
    参数:
    original_data: 原始光谱数据 (samples × wavelengths)
    denoised_data: 去噪后的光谱数据 (samples × wavelengths)
    
    返回:
    pointwise_snr: 每个波长点的信噪比 (dB)
    wavelength_values: 波长值数组
    """
    if len(original_data.shape) != 2:
        raise ValueError("输入数据必须是二维数组 (samples × wavelengths)")
    
    samples, wavelengths = original_data.shape
    
    # 计算噪声（残差）
    noise = original_data - denoised_data  # shape: (samples, wavelengths)
    
    # 对每个波长点计算信噪比
    pointwise_snr = np.zeros(wavelengths)
    
    for i in range(wavelengths):
        # 获取该波长点的所有样本值
        original_vals = original_data[:, i]
        denoised_vals = denoised_data[:, i]
        noise_vals = noise[:, i]
        
        # 计算信号功率（去噪后数据的方差）
        signal_power = np.var(denoised_vals)
        
        # 计算噪声功率（残差的方差）
        noise_power = np.var(noise_vals)
        
        # 计算信噪比（比值形式）
        if noise_power > 1e-12:  # 避免数值不稳定
            snr_linear = signal_power / noise_power
            pointwise_snr[i] = snr_linear
        else:
            pointwise_snr[i] = 1e6  # 设置默认高值
    
    # 生成波长轴
    wavelength_values = np.linspace(400, 1000, wavelengths)
    
    return pointwise_snr, wavelength_values

if __name__ == "__main__":
    # 测试代码
    # 生成测试信号
    x = np.linspace(0, 10, 1000)
    test_signal = np.sin(2*np.pi*x) + 0.5*np.sin(6*np.pi*x) + 0.1*np.random.randn(1000)
    
    # 计算原始信号SNR
    original_snr = calculate_snr(test_signal)
    print(f"原始信号SNR: {original_snr:.2f} dB")
    
    # 应用小波去噪
    denoised_signal = apply_wavelet_denoising(test_signal, levels=6)
    
    # 计算去噪后信号SNR
    denoised_snr = calculate_snr(denoised_signal)
    print(f"去噪后信号SNR: {denoised_snr:.2f} dB")
    print(f"SNR改善: {denoised_snr - original_snr:.2f} dB")