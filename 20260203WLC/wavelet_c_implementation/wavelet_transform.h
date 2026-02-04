#ifndef WAVELET_TRANSFORM_H
#define WAVELET_TRANSFORM_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// 小波变换结构体
typedef struct {
    double* data;           // 输入数据
    int length;             // 数据长度
    int levels;             // 分解层数
    double* coefficients;   // 小波系数
    int* coeff_lengths;     // 各层系数长度
    double* reconstructed;  // 重构数据
} WaveletTransform;

// DB4小波滤波器系数
extern const double db4_low_pass[8];
extern const double db4_high_pass[8];

// 函数声明
WaveletTransform* create_wavelet_transform(double* input_data, int data_length);
void destroy_wavelet_transform(WaveletTransform* wt);

// 小波分解函数
int dwt_db4(WaveletTransform* wt, int levels);
int idwt_db4(WaveletTransform* wt);

// 阈值处理函数
void soft_thresholding(double* coefficients, int length, double threshold);
void hard_thresholding(double* coefficients, int length, double threshold);
double universal_threshold(double* last_level_coeffs, int length);

// 通用小波去噪函数
int wavelet_denoise(double* signal, int length, int levels, double threshold, 
                   void (*threshold_func)(double*, int, double));

// 辅助函数
void print_coefficients(WaveletTransform* wt);
double* read_csv_column(const char* filename, int column_index, int* length);
void write_csv(const char* filename, double* data, int length);
double calculate_snr(double* signal, int length);

#endif // WAVELET_TRANSFORM_H