#include "wavelet_transform.h"
#include <ctype.h>

// 读取CSV文件的指定列
double* read_csv_column(const char* filename, int column_index, int* length) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("无法打开文件: %s\n", filename);
        return NULL;
    }
    
    // 计算行数
    int lines = 0;
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), file)) {
        lines++;
    }
    rewind(file);
    
    // 分配内存
    double* data = (double*)malloc(lines * sizeof(double));
    if (!data) {
        fclose(file);
        return NULL;
    }
    
    int data_index = 0;
    char line[1024];
    
    // 跳过表头行（如果有）
    if (fgets(line, sizeof(line), file)) {
        // 检查是否为数字开头判断是否有表头
        char* token = strtok(line, ",");
        if (token && !isdigit(token[0]) && token[0] != '-' && token[0] != '.') {
            // 有表头，继续读取下一行
            printf("检测到表头，跳过第一行\n");
        } else {
            // 无表头，处理第一行数据
            int col = 0;
            do {
                if (col == column_index) {
                    data[data_index++] = atof(token);
                    break;
                }
                col++;
            } while ((token = strtok(NULL, ",")) != NULL);
        }
    }
    
    // 读取数据行
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        int col = 0;
        
        while (token != NULL && col <= column_index) {
            if (col == column_index) {
                data[data_index++] = atof(token);
                break;
            }
            token = strtok(NULL, ",");
            col++;
        }
    }
    
    fclose(file);
    *length = data_index;
    printf("从 %s 读取了 %d 个数据点\n", filename, *length);
    
    return data;
}

// 写入CSV文件
void write_csv(const char* filename, double* data, int length) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        printf("无法创建文件: %s\n", filename);
        return;
    }
    
    fprintf(file, "Index,Value\n");
    for (int i = 0; i < length; i++) {
        fprintf(file, "%d,%.6f\n", i, data[i]);
    }
    
    fclose(file);
}

// 批量处理多个CSV文件
int batch_process_csv_files(char** filenames, int file_count, int column_index, 
                           int levels, double threshold) {
    printf("=== 批量CSV文件处理 ===\n");
    
    for (int file_idx = 0; file_idx < file_count; file_idx++) {
        printf("\n处理文件 %d/%d: %s\n", file_idx + 1, file_count, filenames[file_idx]);
        
        int length;
        double* signal = read_csv_column(filenames[file_idx], column_index, &length);
        
        if (!signal) {
            printf("  ✗ 读取文件失败\n");
            continue;
        }
        
        if (length < 32) {
            printf("  ⚠ 数据点太少 (%d < 32)，跳过处理\n", length);
            free(signal);
            continue;
        }
        
        // 计算原始SNR
        double original_snr = calculate_snr(signal, length);
        printf("  原始SNR: %.2f\n", original_snr);
        
        // 执行小波去噪
        double* signal_copy = (double*)malloc(length * sizeof(double));
        memcpy(signal_copy, signal, length * sizeof(double));
        
        if (wavelet_denoise(signal_copy, length, levels, threshold, soft_thresholding) == 0) {
            double denoised_snr = calculate_snr(signal_copy, length);
            printf("  去噪后SNR: %.2f\n", denoised_snr);
            printf("  SNR改善: %.2f\n", denoised_snr - original_snr);
            
            // 保存去噪结果
            char output_filename[256];
            snprintf(output_filename, sizeof(output_filename), 
                    "denoised_%s", filenames[file_idx]);
            write_csv(output_filename, signal_copy, length);
            printf("  ✓ 结果保存到: %s\n", output_filename);
        } else {
            printf("  ✗ 去噪处理失败\n");
        }
        
        free(signal);
        free(signal_copy);
    }
    
    return 0;
}