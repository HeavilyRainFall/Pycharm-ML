import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import hashlib

# 页面配置 / Page Configuration
st.set_page_config(
    page_title="Interactive Attention Mechanism 交互式注意力机制",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式 / Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .bilingual-text {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .calculation-step {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #ffc107;
    }
    .button-container {
        display: flex;
        gap: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def generate_unique_key(*args):
    """生成唯一的键 / Generate unique key"""
    key_string = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()[:10]


def initialize_parameters():
    """初始化参数 / Initialize parameters"""
    return {
        'seq_len': 3,
        'd_model': 4,
        'd_k': 2,
        'random_seed': 42
    }


def softmax(x):
    """Softmax函数 / Softmax function"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def plot_interactive_matrix(matrix, title_en, title_zh, colorscale='RdBu', show_values=True):
    """创建交互式矩阵热图 / Create interactive matrix heatmap"""
    if show_values:
        text = [[f"{val:.3f}" for val in row] for row in matrix]
        texttemplate = "%{text}"
    else:
        text = None
        texttemplate = None

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale=colorscale,
        hoverongaps=False,
        text=text,
        texttemplate=texttemplate,
        textfont={"size": 12},
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z:.3f}<extra></extra>"
    ))

    # 双语标题 / Bilingual title
    title = f"{title_en}<br>{title_zh}"

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis_title="Columns / 列",
        yaxis_title="Rows / 行",
        width=400,
        height=400,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


def visualize_matrix_multiplication_step_by_step(A, B, C, operation_name_en, operation_name_zh, unique_suffix):
    """逐步可视化矩阵乘法 / Step-by-step matrix multiplication visualization"""

    # 创建步骤状态 / Create step state
    step_key = f"multiplication_step_{unique_suffix}"
    if step_key not in st.session_state:
        st.session_state[step_key] = 0

    steps = [
        "显示输入矩阵 / Show input matrices",
        "计算第一个元素 / Compute first element",
        "计算第二行 / Compute second row",
        "完整结果 / Complete result"
    ]

    # 步骤控制 / Step control
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"**当前步骤 / Current Step:** {steps[st.session_state[step_key]]}")
    with col2:
        next_key = f"next_{unique_suffix}"
        if st.button("下一步 / Next Step", key=next_key) and st.session_state[step_key] < len(steps) - 1:
            st.session_state[step_key] += 1
            st.rerun()
    with col3:
        prev_key = f"prev_{unique_suffix}"
        if st.button("上一步 / Previous Step", key=prev_key) and st.session_state[step_key] > 0:
            st.session_state[step_key] -= 1
            st.rerun()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[
            f"Matrix A ({A.shape})",
            f"Matrix B ({B.shape})",
            f"Result {operation_name_en} ({C.shape})"
        ],
        horizontal_spacing=0.1
    )

    # 根据步骤显示不同的可视化 / Show different visualization based on step
    current_step = st.session_state[step_key]

    if current_step == 0:
        # 只显示输入矩阵 / Show only input matrices
        fig.add_trace(go.Heatmap(z=A, colorscale='Blues', showscale=False), 1, 1)
        fig.add_trace(go.Heatmap(z=B, colorscale='Greens', showscale=False), 1, 2)
        fig.add_trace(go.Heatmap(z=np.zeros_like(C), colorscale='Reds', showscale=False), 1, 3)

    elif current_step == 1:
        # 计算第一个元素 / Compute first element
        fig.add_trace(go.Heatmap(z=A, colorscale='Blues', showscale=False), 1, 1)
        fig.add_trace(go.Heatmap(z=B, colorscale='Greens', showscale=False), 1, 2)

        # 高亮第一个元素的计算 / Highlight computation of first element
        result_partial = np.zeros_like(C)
        result_partial[0, 0] = C[0, 0]
        fig.add_trace(go.Heatmap(z=result_partial, colorscale='Reds', showscale=False), 1, 3)

        # 显示计算过程 / Show calculation process
        with st.expander("计算细节 / Calculation Details"):
            st.markdown("""
            **计算第一个元素 Q[0,0] / Compute first element Q[0,0]:**
            """)
            calculation_text = ""
            for k in range(A.shape[1]):
                calculation_text += f"Input[0,{k}] × W_Q[{k},0] = {A[0, k]:.3f} × {B[k, 0]:.3f} = {A[0, k] * B[k, 0]:.3f}\n"
            calculation_text += f"**总和 / Sum: {C[0, 0]:.3f}**"
            st.code(calculation_text)

    elif current_step == 2:
        # 计算第二行 / Compute second row
        fig.add_trace(go.Heatmap(z=A, colorscale='Blues', showscale=False), 1, 1)
        fig.add_trace(go.Heatmap(z=B, colorscale='Greens', showscale=False), 1, 2)

        result_partial = np.zeros_like(C)
        result_partial[:2, :] = C[:2, :]  # 显示前两行 / Show first two rows
        fig.add_trace(go.Heatmap(z=result_partial, colorscale='Reds', showscale=False), 1, 3)

    else:
        # 显示完整结果 / Show complete result
        fig.add_trace(go.Heatmap(z=A, colorscale='Blues', showscale=False), 1, 1)
        fig.add_trace(go.Heatmap(z=B, colorscale='Greens', showscale=False), 1, 2)
        fig.add_trace(go.Heatmap(z=C, colorscale='Reds', showscale=False), 1, 3)

    # 添加数值标注 / Add value annotations
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if current_step >= 1 or (i == 0 and j == 0):
                fig.add_annotation(x=j, y=i, text=f"{A[i, j]:.2f}", showarrow=False,
                                   font=dict(color="white" if A[i, j] > np.max(A) / 2 else "black"),
                                   xref="x1", yref="y1")

    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if current_step >= 1:
                fig.add_annotation(x=j, y=i, text=f"{B[i, j]:.2f}", showarrow=False,
                                   font=dict(color="white" if B[i, j] > np.max(B) / 2 else "black"),
                                   xref="x2", yref="y2")

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if (current_step >= 2 and i < 2) or current_step >= 3:
                fig.add_annotation(x=j, y=i, text=f"{C[i, j]:.2f}", showarrow=False,
                                   font=dict(color="white" if C[i, j] > np.max(C) / 2 else "black"),
                                   xref="x3", yref="y3")

    title = f"{operation_name_en}<br>{operation_name_zh}"
    fig.update_layout(
        title=title,
        width=900,
        height=400
    )

    return fig


def animate_dot_product_calculation(input_vector, weight_vector, result_value, description_en, description_zh,
                                    unique_suffix):
    """动画显示点积计算 / Animate dot product calculation"""

    # 使用唯一键来管理状态 / Use unique keys for state management
    step_key = f"animation_step_{unique_suffix}"
    current_k_key = f"current_k_{unique_suffix}"

    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    if current_k_key not in st.session_state:
        st.session_state[current_k_key] = 0

    total_steps = len(input_vector) + 1

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"**动画步骤 / Animation Step:** {st.session_state[step_key] + 1}/{total_steps}")
    with col2:
        next_key = f"anim_next_{unique_suffix}"
        if st.button("下一步 / Next Step", key=next_key) and st.session_state[step_key] < total_steps - 1:
            st.session_state[step_key] += 1
            st.session_state[current_k_key] = min(st.session_state[step_key], len(input_vector) - 1)
            st.rerun()
    with col3:
        reset_key = f"anim_reset_{unique_suffix}"
        if st.button("重置 / Reset", key=reset_key):
            st.session_state[step_key] = 0
            st.session_state[current_k_key] = 0
            st.rerun()

    fig = go.Figure()

    # 显示输入向量 / Show input vector
    positions = list(range(len(input_vector)))

    fig.add_trace(go.Scatter(
        x=positions, y=input_vector,
        mode='markers+lines+text',
        marker=dict(size=15, color='blue'),
        line=dict(color='blue', width=2),
        text=[f"{x:.3f}" for x in input_vector],
        textposition="top center",
        name="Input Vector / 输入向量"
    ))

    # 显示权重向量 / Show weight vector
    fig.add_trace(go.Scatter(
        x=positions, y=weight_vector,
        mode='markers+lines+text',
        marker=dict(size=15, color='green'),
        line=dict(color='green', width=2),
        text=[f"{x:.3f}" for x in weight_vector],
        textposition="bottom center",
        name="Weight Vector / 权重向量"
    ))

    # 根据动画步骤显示计算过程 / Show calculation process based on animation step
    current_sum = 0
    calculation_steps = []

    for k in range(len(input_vector)):
        product = input_vector[k] * weight_vector[k]
        if k <= st.session_state[current_k_key]:
            current_sum += product

            # 高亮当前计算 / Highlight current calculation
            fig.add_annotation(
                x=k, y=(input_vector[k] + weight_vector[k]) / 2,
                text=f"{input_vector[k]:.3f} × {weight_vector[k]:.3f} = {product:.3f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='red',
                bgcolor="yellow" if k == st.session_state[current_k_key] else "lightyellow"
            )

            calculation_steps.append(f"k={k}: {input_vector[k]:.3f} × {weight_vector[k]:.3f} = {product:.3f}")

    # 显示当前总和 / Show current sum
    if st.session_state[step_key] >= len(input_vector):
        current_sum = result_value

    fig.update_layout(
        title=f"{description_en}<br>{description_zh}<br>当前总和 / Current Sum: {current_sum:.4f}",
        xaxis_title="Dimension Index / 维度索引",
        yaxis_title="Value / 值",
        showlegend=True,
        width=600,
        height=400
    )

    # 显示计算步骤 / Show calculation steps
    with st.expander("计算步骤 / Calculation Steps"):
        for step in calculation_steps:
            st.write(step)
        if st.session_state[step_key] >= len(input_vector):
            st.success(f"**最终结果 / Final Result: {result_value:.4f}**")

    return fig


def main():
    """主函数 / Main function"""

    # 标题 / Header
    st.markdown('<h1 class="main-header">🧠 Interactive Attention Mechanism 交互式注意力机制</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="bilingual-text">
    <b>English:</b> This interactive demo demonstrates the <b>Scaled Dot-Product Attention</b> mechanism used in Transformers. 
    Adjust parameters and explore each computation step in detail.<br><br>
    <b>中文:</b> 这个交互式演示展示了Transformer中使用的<b>缩放点积注意力</b>机制。
    调整参数并详细探索每个计算步骤。
    </div>
    """, unsafe_allow_html=True)

    # 侧边栏控制 / Sidebar controls
    st.sidebar.header("🔧 Configuration Parameters 配置参数")

    # 参数控制 / Parameter controls
    seq_len = st.sidebar.slider(
        "Sequence Length 序列长度",
        min_value=2, max_value=5, value=3,
        help="Number of tokens in the input sequence / 输入序列中的标记数量"
    )

    d_model = st.sidebar.slider(
        "Embedding Dimension (d_model) 嵌入维度",
        min_value=2, max_value=6, value=4,
        help="Dimension of input token embeddings / 输入标记嵌入的维度"
    )

    d_k = st.sidebar.slider(
        "Key/Query Dimension (d_k) 键/查询维度",
        min_value=1, max_value=4, value=2,
        help="Dimension of projected Q, K, V matrices / 投影后的Q、K、V矩阵的维度"
    )

    random_seed = st.sidebar.number_input(
        "Random Seed 随机种子",
        min_value=0, max_value=100, value=42,
        help="Seed for reproducible random weights / 可重现随机权重的种子"
    )

    # 初始化参数 / Initialize parameters
    np.random.seed(random_seed)

    # 生成输入嵌入 / Generate input embeddings
    if st.sidebar.checkbox("Use Random Input Embeddings 使用随机输入嵌入", value=False):
        input_embedding = np.random.randn(seq_len, d_model) * 0.5 + 1.0
    else:
        default = np.array([
            [1.2, 0.8, 0.5, 1.0],
            [0.9, 1.1, 0.7, 0.6],
            [0.3, 0.5, 1.3, 0.9]
        ])
        if seq_len <= 3 and d_model <= 4:
            input_embedding = default[:seq_len, :d_model]
        else:
            input_embedding = np.random.randn(seq_len, d_model) * 0.5 + 1.0

    # 步骤1：输入嵌入 / Step 1: Input Embeddings
    st.markdown('<div class="section-header">📥 Step 1: Input Embeddings 输入嵌入</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Input Matrix 输入矩阵**")
        st.write(f"Shape 形状: `{input_embedding.shape}` (sequence_length × embedding_dim / 序列长度 × 嵌入维度)")
        fig_input = plot_interactive_matrix(
            input_embedding,
            "Input Embedding Matrix",
            "输入嵌入矩阵"
        )
        st.plotly_chart(fig_input, use_container_width=True)

    with col2:
        st.markdown("**Explanation 解释**")
        st.info("""
        **English:** Each row represents a token's embedding vector. 
        - **Rows**: Token positions in sequence
        - **Columns**: Feature dimensions in embedding space

        **中文:** 每行代表一个标记的嵌入向量。
        - **行**: 序列中的标记位置
        - **列**: 嵌入空间中的特征维度
        """)

    # 生成权重矩阵 / Generate weight matrices
    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_k) * 0.1

    # 计算Q、K、V / Compute Q, K, V
    Q = np.dot(input_embedding, W_Q)
    K = np.dot(input_embedding, W_K)
    V = np.dot(input_embedding, W_V)

    # 步骤2：Q/K/V投影 / Step 2: Q/K/V Projection
    st.markdown('<div class="section-header">🔑 Step 2: Q/K/V Projection Q/K/V投影</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="bilingual-text">
    <b>English:</b> The input embeddings are projected into three different spaces using learned weight matrices:
    - <b>Q (Query)</b>: What information to look for → `Input × W_Q`
    - <b>K (Key)</b>: What information is available → `Input × W_K`  
    - <b>V (Value)</b>: Actual content to retrieve → `Input × W_V`<br><br>

    <b>中文:</b> 输入嵌入通过学习的权重矩阵投影到三个不同的空间：
    - <b>Q (查询)</b>: 要寻找什么信息 → `输入 × W_Q`
    - <b>K (键)</b>: 可用的信息是什么 → `输入 × W_K`
    - <b>V (值)</b>: 要检索的实际内容 → `输入 × W_V`
    </div>
    """, unsafe_allow_html=True)

    # 交互式矩阵乘法演示 / Interactive matrix multiplication demo
    st.markdown("#### 交互式矩阵乘法演示 / Interactive Matrix Multiplication Demo")

    # 选择要可视化的token和维度 / Select token and dimension to visualize
    col1, col2 = st.columns(2)
    with col1:
        token_idx = st.selectbox("选择标记 / Select token:", range(seq_len), index=0)
    with col2:
        dimension_idx = st.selectbox("选择维度 / Select dimension:", range(d_k), index=0)

    # 生成唯一后缀 / Generate unique suffix
    unique_suffix_q = generate_unique_key("Q", token_idx, dimension_idx)

    # 显示矩阵乘法步骤 / Show matrix multiplication steps
    st.markdown("##### 矩阵乘法步骤 / Matrix Multiplication Steps")
    multiplication_fig = visualize_matrix_multiplication_step_by_step(
        input_embedding, W_Q, Q,
        "Input × W_Q = Q",
        "输入 × W_Q = Q",
        unique_suffix_q
    )
    st.plotly_chart(multiplication_fig, use_container_width=True)

    # 显示点积计算动画 / Show dot product calculation animation
    st.markdown("##### 点积计算动画 / Dot Product Calculation Animation")

    input_vector = input_embedding[token_idx]
    weight_vector = W_Q[:, dimension_idx]
    q_value = np.dot(input_vector, weight_vector)

    # 生成动画的唯一键 / Generate unique key for animation
    anim_suffix = generate_unique_key("anim", token_idx, dimension_idx)

    dot_product_fig = animate_dot_product_calculation(
        input_vector, weight_vector, q_value,
        f"Q[{token_idx},{dimension_idx}] Calculation",
        f"Q[{token_idx},{dimension_idx}] 计算",
        anim_suffix
    )
    st.plotly_chart(dot_product_fig, use_container_width=True)

    # 显示数学公式 / Show mathematical formula
    st.markdown("##### 数学公式 / Mathematical Formula")

    # 修复的LaTeX公式 - 使用正确的变量名 / Fixed LaTeX formula - using correct variable names
    formula = rf"""
    Q[{token_idx},{dimension_idx}] = \sum_{{k=0}}^{{{d_model - 1}}} \text{{Input}}[{token_idx},k] \times W_Q[k,{dimension_idx}]
    """
    st.latex(formula)

    st.markdown(f"""
    <div class="calculation-step">
    <b>计算结果 / Calculation Result:</b> {q_value:.4f}<br>
    <b>验证 / Verification:</b> Q矩阵中对应位置的值 / Value at corresponding position in Q matrix: {Q[token_idx, dimension_idx]:.4f}
    </div>
    """, unsafe_allow_html=True)

    # 显示所有三个投影 / Show all three projections
    st.markdown("#### 所有三个投影 / All Three Projections")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Query Projection: Q = Input × W_Q 查询投影**")
        fig_Q = plot_interactive_matrix(Q, "Query Matrix (Q)", "查询矩阵 (Q)")
        st.plotly_chart(fig_Q, use_container_width=True)
        st.caption(f"Shape 形状: {Q.shape}")

    with col2:
        st.markdown("**Key Projection: K = Input × W_K 键投影**")
        fig_K = plot_interactive_matrix(K, "Key Matrix (K)", "键矩阵 (K)")
        st.plotly_chart(fig_K, use_container_width=True)
        st.caption(f"Shape 形状: {K.shape}")

    with col3:
        st.markdown("**Value Projection: V = Input × W_V 值投影**")
        fig_V = plot_interactive_matrix(V, "Value Matrix (V)", "值矩阵 (V)")
        st.plotly_chart(fig_V, use_container_width=True)
        st.caption(f"Shape 形状: {V.shape}")

    # 继续其他步骤... / Continue with other steps...
    # 步骤3：注意力得分 / Step 3: Attention Scores
    st.markdown('<div class="section-header">📊 Step 3: Attention Scores 注意力得分</div>', unsafe_allow_html=True)

    attention_scores = np.dot(Q, K.T)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Attention Scores 注意力得分**")
        fig_scores = plot_interactive_matrix(
            attention_scores,
            "Attention Scores (Q × Kᵀ)",
            "注意力得分 (Q × Kᵀ)"
        )
        st.plotly_chart(fig_scores, use_container_width=True)
        st.caption(f"Shape 形状: {attention_scores.shape}")

    with col2:
        st.markdown("**Score Interpretation 得分解释**")
        st.info("""
        **English:** Each element (i,j) represents the similarity between:
        - **Query i** (what token i is looking for)
        - **Key j** (what token j can offer)
        Higher values = stronger relationship

        **中文:** 每个元素(i,j)表示以下两者之间的相似性：
        - **查询 i** (标记i正在寻找什么)
        - **键 j** (标记j可以提供什么)
        值越高 = 关系越强
        """)

    # 步骤4：缩放 / Step 4: Scaling
    st.markdown('<div class="section-header">⚖️ Step 4: Scaling 缩放</div>', unsafe_allow_html=True)

    scaled_scores = attention_scores / np.sqrt(d_k)

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_scaled = plot_interactive_matrix(
            scaled_scores,
            f"Scaled Scores (÷ √{d_k} = ÷ {np.sqrt(d_k):.3f})",
            f"缩放后的得分 (÷ √{d_k} = ÷ {np.sqrt(d_k):.3f})"
        )
        st.plotly_chart(fig_scaled, use_container_width=True)

    with col2:
        st.markdown("**Why Scale? 为什么需要缩放?**")
        st.info(f"""
        **English:** Scaling by `1/√d_k` prevents extremely small gradients when `d_k` is large.
        - **d_k** = {d_k}
        - **√d_k** = {np.sqrt(d_k):.3f}
        - **Scale factor** = {1 / np.sqrt(d_k):.3f}
        This maintains stable training in deep networks.

        **中文:** 当`d_k`较大时，通过`1/√d_k`缩放可以防止梯度变得过小。
        - **d_k** = {d_k}
        - **√d_k** = {np.sqrt(d_k):.3f}
        - **缩放因子** = {1 / np.sqrt(d_k):.3f}
        这有助于在深度网络中保持稳定的训练。
        """)

    # 步骤5：Softmax权重 / Step 5: Softmax Weights
    st.markdown('<div class="section-header">🎯 Step 5: Softmax Normalization Softmax归一化</div>',
                unsafe_allow_html=True)

    attention_weights = softmax(scaled_scores)

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_weights = plot_interactive_matrix(
            attention_weights,
            "Attention Weights (Softmax)",
            "注意力权重 (Softmax)",
            colorscale='Blues'
        )
        st.plotly_chart(fig_weights, use_container_width=True)
        st.caption(f"每行求和为1 / Each row sums to 1: {np.sum(attention_weights, axis=1).round(4)}")

    with col2:
        st.markdown("**Weight Interpretation 权重解释**")
        st.info("""
        **English:** Softmax converts scores to probability distribution:
        - Each row shows how a token distributes its attention
        - Values range 0-1, each row sums to 1
        - Higher values = more attention to that position

        **中文:** Softmax将得分转换为概率分布：
        - 每行显示一个标记如何分配其注意力
        - 值范围0-1，每行总和为1
        - 值越高 = 对该位置的注意力越多
        """)

    # 步骤6：最终输出 / Step 6: Final Output
    st.markdown('<div class="section-header">🚀 Step 6: Contextual Embeddings 上下文嵌入</div>', unsafe_allow_html=True)

    attention_output = np.dot(attention_weights, V)

    col1, col2 = st.columns([1, 1])

    with col1:
        fig_output = plot_interactive_matrix(
            attention_output,
            "Final Attention Output",
            "最终注意力输出",
            colorscale='Viridis'
        )
        st.plotly_chart(fig_output, use_container_width=True)
        st.caption(f"Shape 形状: {attention_output.shape}")

    with col2:
        st.markdown("**Output Interpretation 输出解释**")
        st.info("""
        **English:** Each output token is a weighted combination of all value vectors:
        - **Contextual embeddings**: Each token now contains information from relevant tokens
        - **Shape preserved**: Output has same sequence length but different semantic meaning
        - **Foundation for transformers**: This mechanism enables modeling long-range dependencies

        **中文:** 每个输出标记是所有值向量的加权组合：
        - **上下文嵌入**: 每个标记现在包含来自相关标记的信息
        - **形状保持**: 输出具有相同的序列长度但不同的语义含义
        - **Transformer的基础**: 这种机制能够建模长距离依赖关系
        """)

    # 最终总结 / Final Summary
    st.markdown("### 🔄 转换总结 / Transformation Summary")

    summary_data = {
        "步骤 / Step": [
            "输入 / Input",
            "Q投影 / Q Projection",
            "K投影 / K Projection",
            "V投影 / V Projection",
            "得分 / Scores",
            "权重 / Weights",
            "输出 / Output"
        ],
        "操作 / Operation": [
            "原始嵌入 / Raw embeddings",
            "输入 × W_Q / Input × W_Q",
            "输入 × W_K / Input × W_K",
            "输入 × W_V / Input × W_V",
            "Q × Kᵀ / Q × Kᵀ",
            "Softmax(得分/√dₖ) / Softmax(Scores/√dₖ)",
            "权重 × V / Weights × V"
        ],
        "形状 / Shape": [
            f"{input_embedding.shape}",
            f"{Q.shape}",
            f"{K.shape}",
            f"{V.shape}",
            f"{attention_scores.shape}",
            f"{attention_weights.shape}",
            f"{attention_output.shape}"
        ]
    }

    # 显示总结表格 / Show summary table
    st.table(summary_data)


if __name__ == "__main__":
    main()