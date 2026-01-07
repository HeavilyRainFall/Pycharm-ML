# 马氏距离与高斯函数关系可视化项目总结

## 项目概述

我们成功创建了一个教学互动网页应用，用于展示马氏距离的几何意义以及它与椭圆和高斯函数的关系。该应用使用Streamlit框架构建，提供了直观的可视化界面。

## 文件结构

- [app.py](file:///D:/pycharm%20jupyter%20git/mahalanobis_distance_visualization/app.py) - 主Streamlit应用文件，包含所有可视化逻辑和交互功能
- [requirements.txt](file:///D:/pycharm%20git/mahalanobis_distance_visualization/requirements.txt) - 项目依赖包列表
- [README.md](file:///D:/pycharm%20jupyter%20git/mahalanobis_distance_visualization/README.md) - 项目说明文档
- [USAGE.md](file:///D:/pycharm%20jupyter%20git/mahalanobis_distance_visualization/USAGE.md) - 详细使用说明
- [run_app.py](file:///D:/pycharm%20jupyter%20git/mahalanobis_distance_visualization/run_app.py) - 应用启动脚本
- [PROJECT_SUMMARY.md](file:///D:/pycharm%20jupyter%20git/mahalanobis_distance_visualization/PROJECT_SUMMARY.md) - 项目总结文档

## 主要功能

### 1. 马氏距离椭圆可视化
- 可视化马氏距离定义的椭圆等高线
- 显示椭圆的主轴方向
- 交互式调整分布参数（均值、协方差）

### 2. 二维高斯分布可视化
- 显示概率密度分布图
- 颜色深浅表示概率密度大小
- 等高线表示相同概率密度的点

### 3. 交互式计算
- 实时计算任意点与分布均值的马氏距离
- 在图上可视化连接线
- 显示具体的距离数值

### 4. 教学说明
- 马氏距离数学定义
- 几何意义解释
- 与高斯分布关系说明
- 实际应用场景介绍

## 技术特点

1. **交互式参数调整**：用户可以通过侧边栏滑块实时调整分布参数
2. **双视图模式**：可选择椭圆分布视图或高斯分布视图
3. **中文字体支持**：正确显示中文标签和说明
4. **响应式设计**：适配不同屏幕尺寸
5. **错误处理**：对不可逆协方差矩阵等异常情况进行处理

## 数学原理

### 马氏距离定义
$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

其中：
- $x$ 是待测点
- $\mu$ 是分布的均值向量
- $\Sigma$ 是协方差矩阵
- $\Sigma^{-1}$ 是协方差矩阵的逆矩阵

### 与椭圆和高斯分布的关系
1. 等马氏距离的点形成椭圆形状的等高线
2. 二维高斯分布的等概率密度线为椭圆
3. 椭圆的形状和方向由协方差矩阵决定

## 使用方法

1. 安装依赖：`pip install -r requirements.txt`
2. 启动应用：`streamlit run app.py`
3. 或使用启动脚本：`python run_app.py`

## 应用场景

- 教学演示：帮助学生理解马氏距离的几何意义
- 数据分析：可视化数据分布形状
- 异常检测：理解马氏距离在异常检测中的作用

这个应用成功实现了教学互动网页的目标，直观地展示了马氏距离、椭圆与高斯函数之间的深刻联系。