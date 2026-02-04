# C语言小波变换去噪程序

这是一个用C语言实现的光谱小波去噪程序，支持DB4小波基的小波变换和去噪功能。

## 📁 文件结构

```
wavelet_c_implementation/
├── wavelet_transform.h      # 头文件，包含函数声明和结构体定义
├── wavelet_transform.c      # 核心小波变换实现
├── main.c                   # 基本测试程序
├── csv_processor.c          # CSV文件处理功能
├── interactive.c            # 交互式命令行界面
├── interactive_main.c       # 交互式程序入口
├── Makefile                # 编译配置文件
└── README.md               # 本说明文件
```

## 🚀 编译和运行

### Windows (使用MinGW或Visual Studio)
```bash
gcc -o wavelet_test main.c wavelet_transform.c csv_processor.c -lm
gcc -o wavelet_interactive interactive_main.c wavelet_transform.c csv_processor.c interactive.c -lm
```

### Linux/macOS
```bash
make
# 或者
make all
```

## 📊 功能特性

### 核心功能
- ✅ DB4小波基的离散小波变换(DWT)
- ✅ 多层小波分解和重构
- ✅ 软阈值和硬阈值去噪
- ✅ 通用阈值计算方法
- ✅ 信噪比(SNR)计算

### 数据处理
- ✅ CSV文件读取和写入
- ✅ 多列CSV文件批量处理
- ✅ 自动表头检测
- ✅ 交互式命令行界面

### 算法特点
- 使用周期延拓边界处理
- 支持任意长度信号处理
- 内存安全的动态分配
- 重构精度验证

## 💡 使用示例

### 1. 基本测试
```bash
./wavelet_test
```
自动生成测试信号并执行去噪处理

### 2. 交互式使用
```bash
./wavelet_interactive
```

交互式命令示例：
```
wavelet> load data.csv 1
wavelet> process 4 0.1
wavelet> snr
wavelet> save result.csv
wavelet> quit
```

### 3. 编程接口使用
```c
#include "wavelet_transform.h"

// 创建测试信号
double signal[256];
// ... 初始化信号数据 ...

// 执行小波去噪
wavelet_denoise(signal, 256, 4, 0.1, soft_thresholding);

// 计算信噪比
double snr = calculate_snr(signal, 256);
```

## ⚙️ 参数说明

### 小波去噪参数
- **分解层数**: 通常4-6层，影响频率分辨率
- **阈值**: 控制去噪强度，可手动设置或自动计算
- **阈值类型**: 
  - 软阈值：温和收缩，保持信号连续性
  - 硬阈值：直接截断，去噪效果更强

### CSV处理参数
- **列索引**: 从0开始的列号
- **文件格式**: 支持有表头和无表头的CSV文件

## 📈 性能特点

- **时间复杂度**: O(N log N)，其中N为信号长度
- **内存使用**: 约3倍信号长度的额外内存
- **处理速度**: 在现代CPU上可实时处理千点级信号

## 🔧 开发环境要求

- C99或更高标准的C编译器
- 数学库支持(-lm链接选项)
- 标准C库函数支持

## 📝 注意事项

1. 信号长度建议为2的幂次，处理效果最佳
2. 阈值参数需要根据具体应用场景调整
3. CSV文件应使用英文逗号分隔符
4. 程序假设输入数据为双精度浮点数

## 🆘 常见问题

**Q: 编译时出现数学函数未定义错误**
A: 确保链接了数学库：`gcc ... -lm`

**Q: 处理结果不理想**
A: 尝试调整分解层数和阈值参数

**Q: CSV文件读取失败**
A: 检查文件格式和编码，确保使用英文逗号分隔

---
*本程序基于PyWavelets的算法原理实现，保持了良好的去噪效果和计算效率*