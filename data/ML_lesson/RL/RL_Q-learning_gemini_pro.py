import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import random

# ==========================================
# 1. 页面配置与样式
# ==========================================
st.set_page_config(
    page_title="RL 寻宝游戏实验室",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 1rem;}
    h1 {color: #0066cc;}
    .stButton>button {width: 100%; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. 核心类：网格世界环境 (Grid World Environment)
# ==========================================
class GridWorld:
    def __init__(self, size=5, num_traps=3):
        self.size = size
        self.num_traps = num_traps
        # 动作空间: 上, 下, 左, 右
        self.actions = ['↑', '↓', '←', '→']
        self.n_actions = len(self.actions)

        # 初始化位置
        # 如果 session_state 中没有地图信息，则随机生成陷阱
        if 'traps' not in st.session_state:
            self.reset_map()
        else:
            self.traps = st.session_state.traps
            self.goal = st.session_state.goal
            self.agent_pos = st.session_state.agent_pos

    def reset_map(self):
        """重新生成地图布局"""
        self.goal = (self.size - 1, self.size - 1)  # 终点固定在右下角

        # 随机生成陷阱，避免起点和终点
        potential_traps = []
        for r in range(self.size):
            for c in range(self.size):
                if (r, c) != (0, 0) and (r, c) != self.goal:
                    potential_traps.append((r, c))

        self.traps = random.sample(potential_traps, self.num_traps)
        self.agent_pos = (0, 0)  # 起点

        # 保存到 session_state
        st.session_state.traps = self.traps
        st.session_state.goal = self.goal
        st.session_state.agent_pos = self.agent_pos
        # 重置 Q 表和训练记录
        st.session_state.q_table = np.zeros((self.size * self.size, self.n_actions))
        st.session_state.rewards_history = []
        st.session_state.episode_count = 0

    def reset_agent(self):
        """只重置智能体回到起点，不改变地图"""
        self.agent_pos = (0, 0)
        return self.get_state_index()

    def get_state_index(self, pos=None):
        """将二维坐标 (row, col) 转换为一维索引 (0~24)"""
        if pos is None: pos = self.agent_pos
        return pos[0] * self.size + pos[1]

    def step(self, action_idx):
        """执行动作，返回: next_state, reward, done"""
        r, c = self.agent_pos

        # 根据动作移动
        if action_idx == 0:  # 上
            r = max(0, r - 1)
        elif action_idx == 1:  # 下
            r = min(self.size - 1, r + 1)
        elif action_idx == 2:  # 左
            c = max(0, c - 1)
        elif action_idx == 3:  # 右
            c = min(self.size - 1, c + 1)

        next_pos = (r, c)
        self.agent_pos = next_pos
        next_state = self.get_state_index()

        # 判断奖励机制
        if next_pos == self.goal:
            return next_state, 100, True  # 找到宝藏，奖励 +100，结束
        elif next_pos in self.traps:
            return next_state, -100, True  # 掉进陷阱，奖励 -100，结束
        else:
            return next_state, -1, False  # 普通移动，扣 1 分（鼓励走最短路）


# ==========================================
# 3. 初始化 Session State
# ==========================================
if 'env' not in st.session_state:
    env = GridWorld()
    st.session_state.env = env
    st.session_state.q_table = np.zeros((25, 4))  # 25个状态, 4个动作
    st.session_state.rewards_history = []
    st.session_state.episode_count = 0
else:
    env = st.session_state.env

# ==========================================
# 4. 侧边栏：控制台
# ==========================================
st.sidebar.header("🎛️ 控制台 (Control Panel)")

st.sidebar.subheader("1. 超参数 (Hyperparameters)")
epsilon = st.sidebar.slider("探索率 (Epsilon)", 0.0, 1.0, 0.1,
                            help="越高越喜欢乱走(探索)，越低越喜欢按经验走(利用)。")
alpha = st.sidebar.slider("学习率 (Alpha)", 0.01, 1.0, 0.1,
                          help="决定新知识覆盖旧知识的速度。1.0 表示完全只信刚发生的。")
gamma = st.sidebar.slider("折扣因子 (Gamma)", 0.1, 0.99, 0.9,
                          help="越接近 1，智能体越有远见，看重未来的奖励。")

st.sidebar.subheader("2. 操作 (Actions)")


# 训练逻辑函数
def train_agent(episodes, sleep_time=0):
    progress_bar = st.sidebar.progress(0)

    for ep in range(episodes):
        state = env.reset_agent()
        done = False
        total_reward = 0

        while not done:
            # --- Epsilon-Greedy 策略 ---
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(env.n_actions)  # 探索：随机选
            else:
                action = np.argmax(st.session_state.q_table[state])  # 利用：选Q值最大的

            # --- 与环境交互 ---
            next_state, reward, done = env.step(action)

            # --- Q-Learning 核心更新公式 (贝尔曼方程) ---
            # Q_new = Q_old + alpha * (Reward + gamma * max(Q_next) - Q_old)

            old_value = st.session_state.q_table[state, action]
            next_max = np.max(st.session_state.q_table[next_state])

            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            st.session_state.q_table[state, action] = new_value

            state = next_state
            total_reward += reward

        st.session_state.rewards_history.append(total_reward)
        st.session_state.episode_count += 1
        progress_bar.progress((ep + 1) / episodes)

        if sleep_time > 0:
            time.sleep(sleep_time)

    progress_bar.empty()


col_act1, col_act2 = st.sidebar.columns(2)
if col_act1.button("🎲 训练 1 回合"):
    train_agent(1)
    st.sidebar.success("训练完成！观察 Q 表的变化。")

if col_act2.button("🚀 快速训练 500 回合"):
    with st.spinner("正在疯狂训练中..."):
        train_agent(500)
    st.sidebar.success(f"已完成 500 次训练！总计: {st.session_state.episode_count}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 重置环境 (新地图)"):
    env.reset_map()
    st.experimental_rerun()

# ==========================================
# 5. 主界面：可视化
# ==========================================

st.title("🤖 强化学习交互实验室：Q-Learning")
st.markdown("在这个 5x5 的世界里，智能体 (🤖) 需要学会避开陷阱 (🔥) 并找到宝藏 (💰)。")

col1, col2 = st.columns([1, 1])

# --- 左侧：游戏地图 ---
with col1:
    st.subheader("🗺️ 游戏地图 (The World)")

    # 构建用于绘图的数据
    grid_data = []
    annotations = []

    # 绘制背景热力图（为了美观，全白或根据Q值最大值上色）
    # 这里我们用“该状态的最大Q值”来给地图上色，直观显示哪里比较好
    max_q_map = np.max(st.session_state.q_table, axis=1).reshape(5, 5)

    fig = go.Figure(data=go.Heatmap(
        z=max_q_map,
        x=[str(i) for i in range(5)],
        y=[str(i) for i in range(5)],
        colorscale='Blues',
        showscale=True,
        hoverinfo='z',
        name="Max Q-Value"
    ))

    # 添加 Emoji 图标
    for r in range(5):
        for c in range(5):
            icon = ""
            pos = (r, c)
            if pos == st.session_state.agent_pos:
                icon = "🤖"  # 智能体
            elif pos == st.session_state.goal:
                icon = "💰"  # 宝藏
            elif pos in st.session_state.traps:
                icon = "🔥"  # 陷阱

            # 如果是普通格子，显示建议的方向箭头（如果Q值不全为0）
            state_idx = r * 5 + c
            if icon == "" and np.max(st.session_state.q_table[state_idx]) != 0:
                best_act = np.argmax(st.session_state.q_table[state_idx])
                icon = env.actions[best_act]  # 显示箭头

            if icon:
                annotations.append(dict(
                    x=c, y=r, text=icon, showarrow=False,
                    font=dict(size=30, color="black")
                ))

    fig.update_layout(
        width=400, height=400,
        xaxis=dict(side='top', dtick=1),
        yaxis=dict(autorange='reversed', dtick=1),  # y轴反转，符合矩阵直觉
        annotations=annotations,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"当前状态: {st.session_state.agent_pos} | 总训练回合: {st.session_state.episode_count}")

# --- 右侧：Q 表格可视化 ---
with col2:
    st.subheader("🧠 智能体的大脑 (The Q-Table)")
    st.markdown("Q 表记录了：**在某个位置(State)，做某个动作(Action)能得多少分？**")

    # 将 Q 表转换为 DataFrame 以便展示
    df_q = pd.DataFrame(
        st.session_state.q_table,
        columns=["↑ 上", "↓ 下", "← 左", "→ 右"],
        index=[f"位置 {i}" for i in range(25)]
    )

    # 高亮显示当前 Agent 所在的那一行
    current_state_idx = env.get_state_index()


    def highlight_current(s):
        is_current = s.name == f"位置 {current_state_idx}"
        return ['background-color: yellow' if is_current else '' for _ in s]


    # 使用 Pandas Styler 进行热力图着色
    st.dataframe(
        df_q.style.background_gradient(cmap='RdYlGn', axis=None)
        .apply(highlight_current, axis=1)
        .format("{:.2f}"),
        height=400
    )

# --- 底部：训练曲线 ---
st.subheader("📈 训练成效 (Learning Curve)")
if len(st.session_state.rewards_history) > 0:
    # 使用滑动平均来平滑曲线
    rewards = st.session_state.rewards_history
    df_rewards = pd.DataFrame({"Episode": range(len(rewards)), "Reward": rewards})

    # 只有数据足够多时才显示滚动平均
    if len(rewards) > 20:
        df_rewards["Rolling_Avg"] = df_rewards["Reward"].rolling(window=20).mean()
        y_col = ["Reward", "Rolling_Avg"]
    else:
        y_col = ["Reward"]

    fig_chart = px.line(df_rewards, x="Episode", y=y_col,
                        title="每回合获得的总奖励 (Total Reward per Episode)")
    fig_chart.update_layout(yaxis_range=[-150, 120])  # 固定Y轴范围方便观察
    st.plotly_chart(fig_chart, use_container_width=True)
else:
    st.write("暂无训练数据，请点击侧边栏的“训练”按钮。")

# --- 教育性解释 ---
with st.expander("🎓 什么是 Q-Learning? (点击查看原理)"):
    st.markdown(r"""
    **Q-Learning** 是一种让机器通过“试错”来学习的算法。

    它维护一张表格 **Q-Table**，表里的每一个数字 $Q(s, a)$ 代表：
    > 在状态 $s$ (例如位置 0,0) 下，采取动作 $a$ (例如向下走)，**长期来看**能获得多少好处。

    **核心公式 (贝尔曼方程):**
    $$
    Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max Q(s', a') - Q(s,a)]
    $$

    - **$Q(s,a)$**: 旧的经验值。
    - **$\alpha$ (学习率)**: 我们多大程度上相信新的尝试。
    - **$R$ (奖励)**: 这一步立刻拿到的分数（踩到陷阱-100，拿到宝藏+100）。
    - **$\gamma \max Q(s', a')$**: 对未来的预估。虽然这一步只拿了 -1 分，但如果下一步能拿到 +100，那这一步也是好棋。
    """)