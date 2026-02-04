#include "wavelet_transform.h"

// DB4小波滤波器系数 (正交归一化)
const double db4_low_pass[8] = {
    -0.010597401785069032, 0.0328830116668852, 0.030841381835560764,
    -0.18703481171909309, -0.027983769416859854, 0.6308807679298589,
    0.7148465705529157, 0.2303778133088965
};

const double db4_high_pass[8] = {
    0.2303778133088965, -0.7148465705529157, 0.6308807679298589,
    0.027983769416859854, -0.18703481171909309, -0.030841381835560764,
    0.0328830116668852, 0.010597401785069032
};

// 创建小波变换对象
WaveletTransform* create_wavelet_transform(double* input_data, int data_length) {
    WaveletTransform* wt = (WaveletTransform*)malloc(sizeof(WaveletTransform));
    if (!wt) {
        printf("内存分配失败\n");
        return NULL;
    }
    
    wt->data = (double*)malloc(data_length * sizeof(double));
    if (!wt->data) {
        free(wt);
        printf("数据内存分配失败\n");
        return NULL;
    }
    
    memcpy(wt->data, input_data, data_length * sizeof(double));
    wt->length = data_length;
    wt->levels = 0;
    wt->coefficients = NULL;
    wt->coeff_lengths = NULL;
    wt->reconstructed = NULL;
    
    return wt;
}

// 销毁小波变换对象
void destroy_wavelet_transform(WaveletTransform* wt) {
    if (wt) {
        if (wt->data) free(wt->data);
        if (wt->coefficients) free(wt->coefficients);
        if (wt->coeff_lengths) free(wt->coeff_lengths);
        if (wt->reconstructed) free(wt->reconstructed);
        free(wt);
    }
}

// DB4小波分解
int dwt_db4(WaveletTransform* wt, int levels) {
    if (!wt || levels <= 0) {
        printf("无效参数\n");
        return -1;
    }
    
    int current_length = wt->length;
    int total_coeffs = 0;
    
    // 计算总的系数数量和各层长度
    wt->coeff_lengths = (int*)malloc((levels + 2) * sizeof(int));
    wt->coeff_lengths[0] = current_length;  // 原始数据长度
    
    for (int level = 1; level <= levels + 1; level++) {
        current_length = (current_length + 1) / 2;  // 向上取整
        wt->coeff_lengths[level] = current_length;
        if (level <= levels) {
            total_coeffs += current_length * 2;  // 近似和细节系数
        } else {
            total_coeffs += current_length;      // 最后一层只有近似系数
        }
    }
    
    // 分配系数存储空间
    wt->coefficients = (double*)calloc(total_coeffs, sizeof(double));
    if (!wt->coefficients) {
        printf("系数内存分配失败\n");
        return -1;
    }
    
    // 执行多层分解
    double* temp_input = (double*)malloc(wt->length * sizeof(double));
    double* temp_output = (double*)malloc(wt->length * sizeof(double));
    
    if (!temp_input || !temp_output) {
        printf("临时内存分配失败\n");
        if (temp_input) free(temp_input);
        if (temp_output) free(temp_output);
        return -1;
    }
    
    memcpy(temp_input, wt->data, wt->length * sizeof(double));
    int coeff_index = 0;
    current_length = wt->length;
    
    for (int level = 0; level < levels; level++) {
        int output_length = (current_length + 1) / 2;
        
        // 应用滤波器和下采样
        for (int i = 0; i < output_length; i++) {
            double approx = 0.0, detail = 0.0;
            
            for (int j = 0; j < 8; j++) {
                int idx = (2 * i + j - 3) % current_length;
                if (idx < 0) idx += current_length;  // 周期延拓
                
                approx += db4_low_pass[j] * temp_input[idx];
                detail += db4_high_pass[j] * temp_input[idx];
            }
            
            wt->coefficients[coeff_index + i] = approx;           // 近似系数
            wt->coefficients[coeff_index + output_length + i] = detail;  // 细节系数
        }
        
        // 准备下一层输入
        memcpy(temp_input, wt->coefficients + coeff_index, output_length * sizeof(double));
        coeff_index += 2 * output_length;
        current_length = output_length;
    }
    
    // 最后一层近似系数
    memcpy(wt->coefficients + coeff_index, temp_input, current_length * sizeof(double));
    
    wt->levels = levels;
    free(temp_input);
    free(temp_output);
    
    printf("DB4小波分解完成，层数: %d\n", levels);
    return 0;
}

// DB4小波重构
int idwt_db4(WaveletTransform* wt) {
    if (!wt || !wt->coefficients || wt->levels <= 0) {
        printf("无效的小波变换对象\n");
        return -1;
    }
    
    wt->reconstructed = (double*)malloc(wt->length * sizeof(double));
    if (!wt->reconstructed) {
        printf("重构内存分配失败\n");
        return -1;
    }
    
    double* temp_data = (double*)malloc(wt->length * sizeof(double));
    double* temp_output = (double*)malloc(wt->length * sizeof(double));
    
    if (!temp_data || !temp_output) {
        printf("临时内存分配失败\n");
        if (temp_data) free(temp_data);
        if (temp_output) free(temp_output);
        return -1;
    }
    
    // 从最高层开始重构
    int coeff_index = 0;
    for (int i = 0; i <= wt->levels; i++) {
        coeff_index += wt->coeff_lengths[i];
    }
    
    // 获取最后一层近似系数
    coeff_index -= wt->coeff_lengths[wt->levels + 1];
    int current_length = wt->coeff_lengths[wt->levels + 1];
    memcpy(temp_data, wt->coefficients + coeff_index, current_length * sizeof(double));
    
    // 逐层向上重构
    for (int level = wt->levels; level >= 1; level--) {
        int next_length = wt->coeff_lengths[level];
        coeff_index -= 2 * next_length;
        
        // 获取当前层的近似和细节系数
        double* approx = wt->coefficients + coeff_index;
        double* detail = wt->coefficients + coeff_index + next_length;
        
        // 上采样和滤波器重构
        for (int i = 0; i < next_length; i++) {
            temp_output[2 * i] = 0.0;
            temp_output[2 * i + 1] = 0.0;
            
            for (int j = 0; j < 8; j++) {
                int filter_idx = 7 - j;  // 滤波器反转
                int idx = i - (j - 3) / 2;
                
                if (idx >= 0 && idx < current_length) {
                    temp_output[2 * i] += db4_low_pass[filter_idx] * approx[idx];
                    temp_output[2 * i + 1] += db4_high_pass[filter_idx] * detail[idx];
                }
            }
        }
        
        // 合并结果
        for (int i = 0; i < 2 * next_length; i++) {
            if (i < wt->coeff_lengths[level - 1]) {
                temp_data[i] = temp_output[i];
            }
        }
        
        current_length = wt->coeff_lengths[level - 1];
    }
    
    memcpy(wt->reconstructed, temp_data, wt->length * sizeof(double));
    
    free(temp_data);
    free(temp_output);
    
    printf("DB4小波重构完成\n");
    return 0;
}

// 软阈值处理
void soft_thresholding(double* coefficients, int length, double threshold) {
    for (int i = 0; i < length; i++) {
        if (fabs(coefficients[i]) > threshold) {
            if (coefficients[i] > 0) {
                coefficients[i] = coefficients[i] - threshold;
            } else {
                coefficients[i] = coefficients[i] + threshold;
            }
        } else {
            coefficients[i] = 0.0;
        }
    }
}

// 硬阈值处理
void hard_thresholding(double* coefficients, int length, double threshold) {
    for (int i = 0; i < length; i++) {
        if (fabs(coefficients[i]) <= threshold) {
            coefficients[i] = 0.0;
        }
    }
}

// 通用阈值计算（通用阈值法）
double universal_threshold(double* last_level_coeffs, int length) {
    // 计算噪声标准差的估计
    double median_abs = 0.0;
    double* abs_coeffs = (double*)malloc(length * sizeof(double));
    
    for (int i = 0; i < length; i++) {
        abs_coeffs[i] = fabs(last_level_coeffs[i]);
    }
    
    // 简单的中位数近似
    for (int i = 0; i < length - 1; i++) {
        for (int j = 0; j < length - i - 1; j++) {
            if (abs_coeffs[j] > abs_coeffs[j + 1]) {
                double temp = abs_coeffs[j];
                abs_coeffs[j] = abs_coeffs[j + 1];
                abs_coeffs[j + 1] = temp;
            }
        }
    }
    
    median_abs = abs_coeffs[length / 2];
    free(abs_coeffs);
    
    double sigma = median_abs / 0.6745;
    double threshold = sigma * sqrt(2.0 * log((double)length));
    
    return threshold;
}

// 通用小波去噪函数
int wavelet_denoise(double* signal, int length, int levels, double threshold, 
                   void (*threshold_func)(double*, int, double)) {
    
    WaveletTransform* wt = create_wavelet_transform(signal, length);
    if (!wt) return -1;
    
    // 执行小波分解
    if (dwt_db4(wt, levels) != 0) {
        destroy_wavelet_transform(wt);
        return -1;
    }
    
    // 应用阈值处理（对细节系数）
    int coeff_start = 0;
    for (int level = 1; level <= levels; level++) {
        int approx_length = wt->coeff_lengths[level];
        int detail_start = coeff_start + approx_length;
        int detail_length = approx_length;
        
        // 对当前层的细节系数应用阈值
        threshold_func(wt->coefficients + detail_start, detail_length, threshold);
        
        coeff_start += 2 * approx_length;
    }
    
    // 重构信号
    if (idwt_db4(wt) != 0) {
        destroy_wavelet_transform(wt);
        return -1;
    }
    
    // 复制去噪后的结果
    memcpy(signal, wt->reconstructed, length * sizeof(double));
    
    destroy_wavelet_transform(wt);
    return 0;
}

// 打印系数信息
void print_coefficients(WaveletTransform* wt) {
    if (!wt || !wt->coefficients) {
        printf("无系数数据\n");
        return;
    }
    
    printf("\n小波系数信息:\n");
    printf("分解层数: %d\n", wt->levels);
    
    int coeff_index = 0;
    for (int level = 1; level <= wt->levels + 1; level++) {
        if (level <= wt->levels) {
            printf("第%d层 - 近似系数(%d个): ", level, wt->coeff_lengths[level]);
            for (int i = 0; i < fmin(5, wt->coeff_lengths[level]); i++) {
                printf("%.4f ", wt->coefficients[coeff_index + i]);
            }
            printf("\n");
            
            printf("第%d层 - 细节系数(%d个): ", level, wt->coeff_lengths[level]);
            for (int i = 0; i < fmin(5, wt->coeff_lengths[level]); i++) {
                printf("%.4f ", wt->coefficients[coeff_index + wt->coeff_lengths[level] + i]);
            }
            printf("\n");
            
            coeff_index += 2 * wt->coeff_lengths[level];
        } else {
            printf("第%d层 - 最终近似系数(%d个): ", level, wt->coeff_lengths[level]);
            for (int i = 0; i < fmin(5, wt->coeff_lengths[level]); i++) {
                printf("%.4f ", wt->coefficients[coeff_index + i]);
            }
            printf("\n");
        }
    }
}

// 计算信噪比
double calculate_snr(double* signal, int length) {
    if (length < 2) return 0.0;
    
    // 计算信号均值
    double mean = 0.0;
    for (int i = 0; i < length; i++) {
        mean += signal[i];
    }
    mean /= length;
    
    // 计算信号功率和噪声功率
    double signal_power = 0.0, noise_power = 0.0;
    for (int i = 0; i < length - 1; i++) {
        signal_power += pow(signal[i] - mean, 2);
        noise_power += pow(signal[i + 1] - signal[i], 2);  // 相邻差分估计噪声
    }
    
    signal_power /= (length - 1);
    noise_power /= (length - 1);
    
    if (noise_power == 0.0) return 1e6;
    
    return signal_power / noise_power;
}