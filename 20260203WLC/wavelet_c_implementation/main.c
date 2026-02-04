#include "wavelet_transform.h"
#include <time.h>

// 生成测试信号
void generate_test_signal(double* signal, int length) {
    srand((unsigned int)time(NULL));
    
    for (int i = 0; i < length; i++) {
        double x = (double)i / length * 10.0 * M_PI;
        // 生成包含多个频率成分的信号加上噪声
        signal[i] = sin(x) + 0.5 * sin(3.0 * x) + 0.3 * cos(5.0 * x) + 
                   0.1 * ((double)rand() / RAND_MAX - 0.5);
    }
}

// 主测试函数
int main() {
    printf("=== C语言小波变换去噪测试 ===\n\n");
    
    const int signal_length = 256;
    double* original_signal = (double*)malloc(signal_length * sizeof(double));
    double* noisy_signal = (double*)malloc(signal_length * sizeof(double));
    double* denoised_signal = (double*)malloc(signal_length * sizeof(double));
    
    if (!original_signal || !noisy_signal || !denoised_signal) {
        printf("内存分配失败\n");
        return -1;
    }
    
    // 生成测试信号
    generate_test_signal(original_signal, signal_length);
    
    // 添加额外噪声模拟实际测量数据
    srand(12345);  // 固定种子以便重复测试
    for (int i = 0; i < signal_length; i++) {
        noisy_signal[i] = original_signal[i] + 0.2 * ((double)rand() / RAND_MAX - 0.5);
        denoised_signal[i] = noisy_signal[i];  // 复制用于处理
    }
    
    printf("✓ 生成测试信号完成\n");
    printf("  信号长度: %d 点\n", signal_length);
    
    // 计算原始信噪比
    double original_snr = calculate_snr(original_signal, signal_length);
    double noisy_snr = calculate_snr(noisy_signal, signal_length);
    printf("  原始信号SNR: %.2f\n", original_snr);
    printf("  加噪信号SNR: %.2f\n", noisy_snr);
    
    // 执行小波去噪
    printf("\n--- 执行小波去噪 ---\n");
    int levels = 4;
    double threshold = 0.1;  // 可以调整的阈值
    
    printf("  分解层数: %d\n", levels);
    printf("  阈值: %.3f\n", threshold);
    printf("  阈值类型: 软阈值\n");
    
    clock_t start_time = clock();
    
    if (wavelet_denoise(denoised_signal, signal_length, levels, threshold, soft_thresholding) == 0) {
        clock_t end_time = clock();
        double processing_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
        
        printf("✓ 小波去噪完成\n");
        printf("  处理时间: %.4f 秒\n", processing_time);
        
        // 计算去噪后信噪比
        double denoised_snr = calculate_snr(denoised_signal, signal_length);
        printf("  去噪后SNR: %.2f\n", denoised_snr);
        printf("  SNR改善: %.2f\n", denoised_snr - noisy_snr);
        printf("  SNR改善比例: %.1f%%\n", (denoised_snr/noisy_snr - 1.0) * 100);
        
        // 验证重构精度
        WaveletTransform* wt_test = create_wavelet_transform(noisy_signal, signal_length);
        if (wt_test && dwt_db4(wt_test, levels) == 0 && idwt_db4(wt_test) == 0) {
            double reconstruction_error = 0.0;
            for (int i = 0; i < signal_length; i++) {
                reconstruction_error += pow(noisy_signal[i] - wt_test->reconstructed[i], 2);
            }
            reconstruction_error = sqrt(reconstruction_error / signal_length);
            printf("  重构误差: %.6f\n", reconstruction_error);
            destroy_wavelet_transform(wt_test);
        }
        
    } else {
        printf("✗ 小波去噪失败\n");
        free(original_signal);
        free(noisy_signal);
        free(denoised_signal);
        return -1;
    }
    
    // 保存结果到文件
    printf("\n--- 保存结果 ---\n");
    write_csv("original_signal.csv", original_signal, signal_length);
    write_csv("noisy_signal.csv", noisy_signal, signal_length);
    write_csv("denoised_signal.csv", denoised_signal, signal_length);
    printf("✓ 结果已保存到CSV文件\n");
    
    // 清理内存
    free(original_signal);
    free(noisy_signal);
    free(denoised_signal);
    
    printf("\n=== 测试完成 ===\n");
    return 0;
}