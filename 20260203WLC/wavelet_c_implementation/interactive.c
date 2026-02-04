#include "wavelet_transform.h"

// 交互式命令行界面
void interactive_interface() {
    printf("\n=== 光谱小波去噪交互式界面 ===\n");
    printf("支持的命令:\n");
    printf("  load <filename> <column>  - 加载CSV文件的指定列\n");
    printf("  process <levels> <threshold> - 执行小波去噪\n");
    printf("  snr                      - 计算当前信号的信噪比\n");
    printf("  save <filename>          - 保存当前结果\n");
    printf("  batch <files...>         - 批量处理多个文件\n");
    printf("  help                     - 显示帮助\n");
    printf("  quit                     - 退出程序\n");
    printf("\n");
    
    double* current_signal = NULL;
    int signal_length = 0;
    double* processed_signal = NULL;
    
    char command[256];
    while (1) {
        printf("wavelet> ");
        if (!fgets(command, sizeof(command), stdin)) break;
        
        // 移除换行符
        command[strcspn(command, "\n")] = 0;
        
        if (strncmp(command, "load", 4) == 0) {
            char filename[256];
            int column;
            if (sscanf(command, "load %s %d", filename, &column) == 2) {
                if (current_signal) free(current_signal);
                current_signal = read_csv_column(filename, column, &signal_length);
                if (current_signal) {
                    printf("✓ 成功加载 %s 第%d列，共%d个数据点\n", 
                           filename, column, signal_length);
                }
            } else {
                printf("用法: load <filename> <column>\n");
            }
        }
        else if (strncmp(command, "process", 7) == 0) {
            if (!current_signal) {
                printf("请先加载信号数据\n");
                continue;
            }
            
            int levels;
            double threshold;
            if (sscanf(command, "process %d %lf", &levels, &threshold) == 2) {
                if (processed_signal) free(processed_signal);
                processed_signal = (double*)malloc(signal_length * sizeof(double));
                memcpy(processed_signal, current_signal, signal_length * sizeof(double));
                
                if (wavelet_denoise(processed_signal, signal_length, levels, threshold, soft_thresholding) == 0) {
                    printf("✓ 小波去噪完成\n");
                    double orig_snr = calculate_snr(current_signal, signal_length);
                    double proc_snr = calculate_snr(processed_signal, signal_length);
                    printf("  原始SNR: %.2f -> 去噪后SNR: %.2f\n", orig_snr, proc_snr);
                } else {
                    printf("✗ 去噪失败\n");
                    free(processed_signal);
                    processed_signal = NULL;
                }
            } else {
                printf("用法: process <levels> <threshold>\n");
            }
        }
        else if (strcmp(command, "snr") == 0) {
            if (processed_signal) {
                double snr = calculate_snr(processed_signal, signal_length);
                printf("当前信号SNR: %.2f\n", snr);
            } else if (current_signal) {
                double snr = calculate_snr(current_signal, signal_length);
                printf("当前信号SNR: %.2f\n", snr);
            } else {
                printf("没有加载信号数据\n");
            }
        }
        else if (strncmp(command, "save", 4) == 0) {
            char filename[256];
            if (sscanf(command, "save %s", filename) == 1) {
                double* data_to_save = processed_signal ? processed_signal : current_signal;
                if (data_to_save) {
                    write_csv(filename, data_to_save, signal_length);
                    printf("✓ 数据已保存到 %s\n", filename);
                } else {
                    printf("没有可保存的数据\n");
                }
            } else {
                printf("用法: save <filename>\n");
            }
        }
        else if (strcmp(command, "help") == 0) {
            printf("支持的命令:\n");
            printf("  load <filename> <column>  - 加载CSV文件的指定列\n");
            printf("  process <levels> <threshold> - 执行小波去噪\n");
            printf("  snr                      - 计算当前信号的信噪比\n");
            printf("  save <filename>          - 保存当前结果\n");
            printf("  batch <files...>         - 批量处理多个文件\n");
            printf("  help                     - 显示帮助\n");
            printf("  quit                     - 退出程序\n");
        }
        else if (strcmp(command, "quit") == 0 || strcmp(command, "exit") == 0) {
            break;
        }
        else if (strlen(command) > 0) {
            printf("未知命令: %s\n", command);
            printf("输入 'help' 查看可用命令\n");
        }
    }
    
    if (current_signal) free(current_signal);
    if (processed_signal) free(processed_signal);
}