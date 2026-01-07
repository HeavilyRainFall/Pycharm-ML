import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from scipy.stats import multivariate_normal
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置页面标题
st.set_page_config(page_title="马氏距离与高斯函数关系可视化", layout="wide")
st.title("马氏距离与高斯函数关系可视化")
st.markdown("""
## 马氏距离的几何意义与椭圆、高斯函数的关系

本应用将帮助您理解：
- 马氏距离的几何意义
- 马氏距离如何定义椭圆等高线
- 椭圆与二维高斯分布的关系
""")

# 侧边栏参数设置
st.sidebar.header("参数设置")
distribution_type = st.sidebar.selectbox("选择分布类型", ["椭圆分布", "高斯分布"])
mean_x = st.sidebar.slider("均值 x", -5.0, 5.0, 0.0, step=0.1)
mean_y = st.sidebar.slider("均值 y", -5.0, 5.0, 0.0, step=0.1)
cov_xx = st.sidebar.slider("协方差 σxx", 0.1, 5.0, 1.0, step=0.1)
cov_yy = st.sidebar.slider("协方差 σyy", 0.1, 5.0, 1.0, step=0.1)
cov_xy = st.sidebar.slider("协方差 σxy", -2.0, 2.0, 0.0, step=0.1)

# 创建协方差矩阵
cov_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

# 计算特征值和特征向量以绘制椭圆
eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

# 生成数据点
x = np.linspace(-6, 6, 300)
y = np.linspace(-6, 6, 300)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
mean = [mean_x, mean_y]

# 计算多变量正态分布
rv = multivariate_normal(mean, cov_matrix)
Z = rv.pdf(pos)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8))

if distribution_type == "椭圆分布":
    # 绘制椭圆等高线
    ax.contour(X, Y, Z, levels=10, alpha=0.6)
    # 绘制主要椭圆轮廓
    ax.contour(X, Y, Z, levels=[0.05, 0.1, 0.15], colors='red', linewidths=2)
    
    # 绘制主轴方向
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width = 2 * np.sqrt(eigenvals[0])
    height = 2 * np.sqrt(eigenvals[1])
    
    ellipse = Ellipse(xy=mean, width=width*2, height=height*2, angle=angle, 
                      facecolor='none', edgecolor='blue', linestyle='--', linewidth=2)
    ax.add_patch(ellipse)
    
    # 绘制均值点
    ax.plot(mean[0], mean[1], 'ro', markersize=10, label='均值点')
    
    ax.set_title("马氏距离椭圆等高线图\n椭圆表示等距离轮廓")
    ax.set_xlabel("X轴")
    ax.set_ylabel("Y轴")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axis('equal')

else:  # 高斯分布
    # 绘制3D高斯分布
    ax.contour(X, Y, Z, levels=20, alpha=0.6)
    contourf = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    
    # 添加颜色条
    fig.colorbar(contourf, ax=ax)
    
    # 绘制均值点
    ax.plot(mean[0], mean[1], 'ro', markersize=10, label='均值点')
    
    ax.set_title("二维高斯分布\n颜色深浅表示概率密度")
    ax.set_xlabel("X轴")
    ax.set_ylabel("Y轴")
    ax.grid(True, alpha=0.3)
    ax.legend()

st.pyplot(fig)

# 解释马氏距离
st.header("马氏距离概念解释")
st.markdown("""
### 什么是马氏距离？

马氏距离（Mahalanobis Distance）是一种度量数据点与数据分布之间距离的方法，定义为：

$$D_M(x) = \\sqrt{(x - \\mu)^T \\Sigma^{-1} (x - \\mu)}$$

其中：
- $x$ 是待测点
- $\\mu$ 是分布的均值向量
- $\\Sigma$ 是协方差矩阵
- $\\Sigma^{-1}$ 是协方差矩阵的逆矩阵

### 马氏距离的几何意义

1. **椭圆等高线**：在二维空间中，所有与均值点具有相同马氏距离的点构成一个椭圆
2. **考虑数据分布**：马氏距离考虑了数据的协方差结构，因此它对数据的尺度和方向敏感
3. **标准化距离**：马氏距离本质上是将数据标准化后计算的欧氏距离

### 与高斯分布的关系

在二维高斯分布中：
- 等概率密度的点形成椭圆形状的等高线
- 这些椭圆的中心位于均值点
- 椭圆的形状和方向由协方差矩阵决定
- 马氏距离相等的点对应相同的概率密度值

""")

# 添加交互式马氏距离计算
st.header("马氏距离计算示例")
st.markdown("""
在下方输入一个点的坐标，计算它与当前分布均值的马氏距离：
""")

col1, col2 = st.columns(2)
with col1:
    point_x = st.number_input("点的 X 坐标", value=1.0, step=0.1)
with col2:
    point_y = st.number_input("点的 Y 坐标", value=1.0, step=0.1)

point = np.array([point_x, point_y])
mean_array = np.array([mean_x, mean_y])

# 计算马氏距离
diff = point - mean_array
try:
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    mahal_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
    
    st.write(f"**计算结果**：")
    st.write(f"点 ({point_x}, {point_y}) 与均值 ({mean_x}, {mean_y}) 的马氏距离为：**{mahal_dist:.4f}**")
    
    # 在图上标记这个点
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.contour(X, Y, Z, levels=10, alpha=0.6)
    contourf = ax2.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.7)
    ax2.plot(mean[0], mean[1], 'ro', markersize=10, label='均值点')
    ax2.plot(point[0], point[1], 'bo', markersize=10, label=f'目标点 ({point_x}, {point_y})')
    
    # 连接均值点和目标点
    ax2.plot([mean[0], point[0]], [mean[1], point[1]], 'b--', alpha=0.7, label=f'马氏距离 = {mahal_dist:.4f}')
    
    fig2.colorbar(contourf, ax=ax2)
    ax2.set_title("马氏距离可视化")
    ax2.set_xlabel("X轴")
    ax2.set_ylabel("Y轴")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axis('equal')
    
    st.pyplot(fig2)
    
except np.linalg.LinAlgError:
    st.error("协方差矩阵不可逆，无法计算马氏距离！请检查参数设置。")

st.markdown("""
### 应用场景

马氏距离在以下领域有广泛应用：
- **异常检测**：识别远离正常数据分布的异常点
- **分类算法**：作为距离度量用于分类
- **聚类分析**：在聚类算法中考虑数据分布形状
- **质量控制**：在工业生产中检测异常产品

调整侧边栏的参数，观察不同协方差矩阵如何影响椭圆形状和高斯分布形态！
""")