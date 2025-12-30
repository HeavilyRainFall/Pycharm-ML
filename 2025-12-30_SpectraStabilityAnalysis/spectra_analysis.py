import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import os
from pathlib import Path

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义数据路径
DATA_PATH = Path("SpectraData")

def load_spectra_data():
    """加载所有光谱数据文件"""
    data_files = list(DATA_PATH.glob("*.csv"))
    spectra_data = {}
    
    for file in data_files:
        # 提取时间信息（从文件名中提取数字部分）
        filename = file.name
        # 找到"-"后和".csv"前的数字
        if " - " in filename:
            time_str = filename.split(" - ")[1].split(".csv")[0]
        else:
            time_str = filename.split("-")[1].split(".csv")[0]
        time_min = int(time_str)
        
        # 读取CSV数据，尝试不同编码
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding='gbk')
        spectra_data[time_min] = df
        
    return spectra_data

def calculate_stability_metrics(ref_data, test_data):
    """计算稳定性指标"""
    # 确保两个数据集的波长相同
    if not ref_data.iloc[:, 0].equals(test_data.iloc[:, 0]):
        # 如果波长不完全相同，进行插值对齐
        common_wavelengths = ref_data.iloc[:, 0]
        test_interp = np.interp(common_wavelengths, test_data.iloc[:, 0], test_data.iloc[:, 1])
    else:
        common_wavelengths = ref_data.iloc[:, 0]
        test_interp = test_data.iloc[:, 1]
    
    ref_intensity = ref_data.iloc[:, 1]
    
    # 计算差异
    intensity_diff = test_interp - ref_intensity
    relative_diff = (intensity_diff / ref_intensity) * 100  # 百分比差异
    
    # 计算稳定性指标
    metrics = {
        'mean_abs_diff': np.mean(np.abs(intensity_diff)),
        'std_diff': np.std(intensity_diff),
        'max_diff': np.max(np.abs(intensity_diff)),
        'mean_rel_diff': np.mean(np.abs(relative_diff)),
        'std_rel_diff': np.std(relative_diff),
        'max_rel_diff': np.max(np.abs(relative_diff)),
        'correlation': np.corrcoef(ref_intensity, test_interp)[0, 1],
        'intensity_diff': intensity_diff,  # 保存原始差异数据
        'relative_diff': relative_diff,    # 保存原始相对差异数据
        'wavelengths': common_wavelengths  # 保存波长数据
    }
    
    return metrics

def create_interactive_plot(spectra_data):
    """创建交互式光谱对比图"""
    # 获取参考数据（0分钟）
    ref_data = spectra_data[0]
    ref_wavelengths = ref_data.iloc[:, 0]
    ref_intensity = ref_data.iloc[:, 1]
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('光谱强度对比', '相对差异 (%)'),
        vertical_spacing=0.1
    )
    
    # 添加参考光谱
    fig.add_trace(
        go.Scatter(
            x=ref_wavelengths,
            y=ref_intensity,
            mode='lines',
            name='参考光谱 (0分钟)',
            line=dict(color='black', width=2)
        ),
        row=1, col=1
    )
    
    # 添加其他时间点的光谱
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    time_points = sorted([t for t in spectra_data.keys() if t != 0])
    
    for i, time_min in enumerate(time_points):
        test_data = spectra_data[time_min]
        # 插值到参考波长轴
        test_interp = np.interp(ref_wavelengths, test_data.iloc[:, 0], test_data.iloc[:, 1])
        
        fig.add_trace(
            go.Scatter(
                x=ref_wavelengths,
                y=test_interp,
                mode='lines',
                name=f'{time_min}分钟',
                line=dict(color=colors[i % len(colors)])
            ),
            row=1, col=1
        )
        
        # 计算相对差异
        relative_diff = ((test_interp - ref_intensity) / ref_intensity) * 100
        
        fig.add_trace(
            go.Scatter(
                x=ref_wavelengths,
                y=relative_diff,
                mode='lines',
                name=f'{time_min}分钟差异',
                line=dict(color=colors[i % len(colors)])
            ),
            row=2, col=1
        )
    
    # 设置图形属性
    fig.update_xaxes(title_text="波长 (nm)", row=1, col=1)
    fig.update_yaxes(title_text="强度", row=1, col=1)
    fig.update_xaxes(title_text="波长 (nm)", row=2, col=1)
    fig.update_yaxes(title_text="相对差异 (%)", row=2, col=1)
    
    fig.update_layout(
        title_text="光源光谱稳定性分析",
        height=800
    )
    
    return fig

def main():
    st.title("光源光谱稳定性分析报告")
    
    # 加载数据
    with st.spinner("正在加载光谱数据..."):
        spectra_data = load_spectra_data()
    
    st.success(f"成功加载 {len(spectra_data)} 个光谱数据文件")
    
    # 显示数据概览
    st.subheader("数据概览")
    time_points = sorted(spectra_data.keys())
    st.write(f"时间点: {time_points} 分钟")
    
    # 选择要分析的时间点
    selected_time = st.selectbox("选择要分析的时间点（分钟）", options=[t for t in time_points if t != 0])
    
    if selected_time:
        # 计算稳定性指标
        ref_data = spectra_data[0]
        test_data = spectra_data[selected_time]
        metrics = calculate_stability_metrics(ref_data, test_data)
        
        # 显示稳定性指标
        st.subheader(f"{selected_time}分钟时的稳定性指标")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("平均绝对差异", f"{metrics['mean_abs_diff']:.4f}")
            st.metric("最大差异", f"{metrics['max_diff']:.4f}")
        with col2:
            st.metric("平均相对差异 (%)", f"{metrics['mean_rel_diff']:.4f}%")
            st.metric("最大相对差异 (%)", f"{metrics['max_rel_diff']:.4f}%")
        with col3:
            st.metric("标准差", f"{metrics['std_diff']:.4f}")
            st.metric("相关系数", f"{metrics['correlation']:.4f}")
        
        # 创建并显示交互式图表
        st.subheader("光谱对比图")
        fig = create_interactive_plot(spectra_data)
        st.plotly_chart(fig, width='stretch')
        
        # 显示详细数据表格
        st.subheader(f"详细数据 - {selected_time}分钟 vs 参考光谱")
        comparison_df = pd.DataFrame({
            '波长 (nm)': metrics['wavelengths'],
            '参考强度 (0分钟)': ref_data.iloc[:, 1],
            f'{selected_time}分钟强度': np.interp(metrics['wavelengths'], test_data.iloc[:, 0], test_data.iloc[:, 1]),
            '强度差异': metrics['intensity_diff'],
            '相对差异 (%)': metrics['relative_diff']
        })
        st.dataframe(comparison_df, height=400)
    
    # 生成整体稳定性报告
    st.subheader("整体稳定性评估")
    all_metrics = {}
    for time_min in [t for t in time_points if t != 0]:
        test_data = spectra_data[time_min]
        metrics = calculate_stability_metrics(ref_data, test_data)
        all_metrics[time_min] = metrics
    
    # 创建稳定性指标随时间变化的图表
    if all_metrics:
        time_values = list(all_metrics.keys())
        mean_diff_values = [all_metrics[t]['mean_abs_diff'] for t in time_values]
        max_diff_values = [all_metrics[t]['max_diff'] for t in time_values]
        corr_values = [all_metrics[t]['correlation'] for t in time_values]
        
        stability_fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('平均绝对差异', '最大差异', '相关系数'),
            vertical_spacing=0.1
        )
        
        stability_fig.add_trace(
            go.Scatter(
                x=time_values,
                y=mean_diff_values,
                mode='lines+markers',
                name='平均绝对差异'
            ),
            row=1, col=1
        )
        
        stability_fig.add_trace(
            go.Scatter(
                x=time_values,
                y=max_diff_values,
                mode='lines+markers',
                name='最大差异'
            ),
            row=2, col=1
        )
        
        stability_fig.add_trace(
            go.Scatter(
                x=time_values,
                y=corr_values,
                mode='lines+markers',
                name='相关系数'
            ),
            row=3, col=1
        )
        
        stability_fig.update_xaxes(title_text="时间 (分钟)", row=3, col=1)
        stability_fig.update_yaxes(title_text="平均差异", row=1, col=1)
        stability_fig.update_yaxes(title_text="最大差异", row=2, col=1)
        stability_fig.update_yaxes(title_text="相关系数", row=3, col=1)
        
        stability_fig.update_layout(
            title_text="光谱稳定性随时间变化",
            height=900
        )
        
        st.plotly_chart(stability_fig, width='stretch')
    
    # 时间序列趋势分析
    st.subheader("时间序列趋势分析")
    if all_metrics:
        time_values = list(all_metrics.keys())
        mean_rel_diff_values = [all_metrics[t]['mean_rel_diff'] for t in time_values]
        max_rel_diff_values = [all_metrics[t]['max_rel_diff'] for t in time_values]
        correlation_values = [all_metrics[t]['correlation'] for t in time_values]
        
        # 创建趋势图
        trend_fig = go.Figure()
        
        trend_fig.add_trace(go.Scatter(
            x=time_values, y=mean_rel_diff_values,
            mode='lines+markers', name='平均相对差异 (%)',
            line=dict(color='blue')
        ))
        
        trend_fig.add_trace(go.Scatter(
            x=time_values, y=max_rel_diff_values,
            mode='lines+markers', name='最大相对差异 (%)',
            line=dict(color='red')
        ))
        
        trend_fig.update_layout(
            title="光谱差异随时间变化趋势",
            xaxis_title="时间 (分钟)",
            yaxis_title="相对差异 (%)",
            width=800,
            height=500
        )
        
        st.plotly_chart(trend_fig, width='stretch')
        
        # 生成定性结论
        st.subheader("定性结论")
        generate_qualitative_assessment(time_values, mean_rel_diff_values, max_rel_diff_values, correlation_values)

def generate_qualitative_assessment(time_values, mean_rel_diff_values, max_rel_diff_values, correlation_values):
    """生成定性评估结论"""
    # 计算关键指标
    avg_mean_rel_diff = np.mean(mean_rel_diff_values)
    avg_max_rel_diff = np.mean(max_rel_diff_values)
    final_correlation = correlation_values[-1] if correlation_values else 1.0
    
    # 计算趋势
    mean_diff_trend = np.polyfit(time_values, mean_rel_diff_values, 1)[0] if len(time_values) > 1 else 0
    max_diff_trend = np.polyfit(time_values, max_rel_diff_values, 1)[0] if len(time_values) > 1 else 0
    
    # 生成评估
    stability_level = ""
    if avg_mean_rel_diff < 2.0:
        stability_level = "优秀"
    elif avg_mean_rel_diff < 5.0:
        stability_level = "良好"
    elif avg_mean_rel_diff < 10.0:
        stability_level = "一般"
    else:
        stability_level = "较差"
    
    # 趋势评估
    trend_desc = ""
    if mean_diff_trend > 0.1:
        trend_desc = "随着时间增加，差异显著增大，光源稳定性下降明显"
    elif mean_diff_trend > 0.05:
        trend_desc = "随着时间增加，差异缓慢增大，光源稳定性略有下降"
    elif mean_diff_trend < -0.05:
        trend_desc = "随着时间增加，差异减小，光源稳定性有所改善"
    else:
        trend_desc = "随着时间增加，差异基本稳定，光源稳定性保持一致"
    
    # 生成结论
    conclusion = f"""
    ## 光源稳定性评估结论
    
    - **整体稳定性水平**: {stability_level}
    - **平均相对差异**: {avg_mean_rel_diff:.2f}%
    - **平均最大相对差异**: {avg_max_rel_diff:.2f}%
    - **最终相关系数**: {final_correlation:.4f}
    - **稳定性趋势**: {trend_desc}
    
    ## 详细分析
    
    1. **稳定性水平**：根据平均相对差异指标，光源的整体稳定性水平为{stability_level}。若平均相对差异小于2%，表示光源非常稳定；若在2%-5%之间，表示光源基本稳定；若在5%-10%之间，表示光源稳定性一般；若超过10%，表示光源稳定性较差。
    
    2. **时间趋势**：从时间序列趋势图可以看出，光源在不同时间点的光谱与初始参考光谱的差异变化情况。{trend_desc}。
    
    3. **峰值差异**：最大相对差异反映了光谱中某些特定波长处的最大变化，该值的大小直接影响光源的适用性。
    
    4. **相关性**：相关系数接近1表示光谱形状保持良好，即使强度有变化，但光谱形状相似度高。
    
    ## 建议
    
    - 若稳定性水平为"优秀"或"良好"：该光源可满足大部分应用需求。
    - 若稳定性水平为"一般"：建议在对稳定性要求不高的场景中使用，或定期校准。
    - 若稳定性水平为"较差"：建议更换更稳定的光源或增加稳定化措施。
    """
    
    st.markdown(conclusion)
    
    # 提供有力论据
    st.subheader("关键论据")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("平均相对差异", f"{avg_mean_rel_diff:.2f}%")
        st.metric("最大相对差异趋势", f"{mean_diff_trend:.4f}/分钟")
    with col2:
        st.metric("最终相关系数", f"{final_correlation:.4f}")
        st.metric("平均最大差异", f"{avg_max_rel_diff:.2f}%")

if __name__ == "__main__":
    main()