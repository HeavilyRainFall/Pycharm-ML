import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置页面标题
st.title("Q-Learning vs SARSA 算法对比可视化演示")

# 创建侧边栏用于参数调整
st.sidebar.header("参数设置")
learning_rate = st.sidebar.slider("学习速率 (α)", 0.1, 1.0, 0.5, 0.1)
exploration_rate = st.sidebar.slider("探索率 (ε)", 0.0, 1.0, 0.3, 0.1)
discount_factor = st.sidebar.slider("折扣因子 (γ)", 0.0, 1.0, 0.9, 0.1)
num_episodes = st.sidebar.slider("训练轮数", 10, 100, 50, 10)

def initialize_q_table(rows, cols, actions):
    """初始化Q表"""
    return np.zeros((rows, cols, len(actions)))

def get_action(state, q_table, exploration_rate, actions):
    """根据ε-greedy策略选择动作"""
    if random.random() < exploration_rate:
        return random.choice(actions)
    else:
        row, col = state
        return actions[np.argmax(q_table[row, col])]

def calculate_reward(state, goal, next_state, obstacle_positions=None):
    """计算奖励函数"""
    if next_state == goal:
        return 100  # 到达目标给予高奖励
    elif state == next_state:
        return -10  # 撞墙惩罚
    else:
        # 使用曼哈顿距离作为奖励的一部分，越靠近目标奖励越高
        distance_to_goal = abs(next_state[0] - goal[0]) + abs(next_state[1] - goal[1])
        return -distance_to_goal  # 距离越远惩罚越大

def update_q_value_q_learning(q_table, state, action, reward, next_state, learning_rate, discount_factor=0.9, actions=None):
    """更新Q值 - Q-Learning算法"""
    row, col = state
    next_row, next_col = next_state
    if actions is None:
        actions = ['up', 'down', 'left', 'right']
    action_idx = actions.index(action)
    
    # Q-learning公式: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
    current_q = q_table[row, col, action_idx]
    max_next_q = np.max(q_table[next_row, next_col])
    new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
    q_table[row, col, action_idx] = new_q
    
    return q_table

def update_q_value_sarsa(q_table, state, action, reward, next_state, next_action, learning_rate, discount_factor=0.9, actions=None):
    """更新Q值 - SARSA算法"""
    row, col = state
    next_row, next_col = next_state
    if actions is None:
        actions = ['up', 'down', 'left', 'right']
    action_idx = actions.index(action)
    next_action_idx = actions.index(next_action)
    
    # SARSA公式: Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
    # 注意：这里使用实际选择的下一个动作next_action，而不是最大Q值对应的动作
    current_q = q_table[row, col, action_idx]
    next_q = q_table[next_row, next_col, next_action_idx]
    new_q = current_q + learning_rate * (reward + discount_factor * next_q - current_q)
    q_table[row, col, action_idx] = new_q
    
    return q_table

def move_agent(state, action, rows, cols):
    """根据动作移动智能体"""
    row, col = state
    if action == 'up' and row > 0:
        return (row - 1, col)
    elif action == 'down' and row < rows - 1:
        return (row + 1, col)
    elif action == 'left' and col > 0:
        return (row, col - 1)
    elif action == 'right' and col < cols - 1:
        return (row, col + 1)
    return state  # 无法移动时保持原位置

def is_goal_reached(state, goal):
    """检查是否到达目标"""
    return state == goal

def visualize_grid_world(grid_size, agent_pos, goal_pos, q_table, episode, step, algorithm_name):
    """可视化网格世界"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建颜色映射
    cmap = ListedColormap(['white', 'lightblue', 'green'])
    
    # 创建网格
    grid = np.zeros((grid_size, grid_size))
    
    # 标记目标位置
    grid[goal_pos[0], goal_pos[1]] = 2
    
    # 标记智能体位置
    grid[agent_pos[0], agent_pos[1]] = 1
    
    # 绘制网格
    ax.imshow(grid, cmap=cmap, interpolation='nearest')
    
    # 添加坐标轴标签
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels(range(grid_size))
    ax.set_yticklabels(range(grid_size))
    
    # 添加网格线
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    
    # 添加标题
    ax.set_title(f"{algorithm_name} - 第 {episode+1} 轮, 第 {step+1} 步\nQ值更新动画", fontsize=14, fontweight='bold')
    
    # 在每个格子上显示Q值
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) != goal_pos:
                q_values = q_table[i, j]
                # 使用不同颜色表示Q值大小
                max_q = np.max(q_values)
                min_q = np.min(q_values)
                if max_q > min_q:
                    normalized_q = (q_values - min_q) / (max_q - min_q)
                else:
                    normalized_q = np.zeros_like(q_values)
                
                # 创建文本
                text = f"U:{q_values[0]:.2f}\nD:{q_values[1]:.2f}\nL:{q_values[2]:.2f}\nR:{q_values[3]:.2f}"
                
                # 设置文本颜色
                colors = ['red', 'blue', 'green', 'purple']
                for k, color in enumerate(colors):
                    if q_values[k] == max_q and max_q > 0:
                        text = text.replace(f"{q_values[k]:.2f}", f"**{q_values[k]:.2f}**")
                        
                ax.text(j, i, text, ha="center", va="center", fontsize=8, fontweight='bold', color='black')
    
    # 显示智能体位置
    ax.plot(agent_pos[1], agent_pos[0], 'ro', markersize=15, label='智能体', zorder=5)
    ax.plot(goal_pos[1], goal_pos[0], 'go', markersize=15, label='目标', zorder=5)
    ax.legend(loc='upper right')  # 移除可能导致错误的prop参数
    
    # 添加说明文字
    ax.text(0.02, 0.98, f"当前状态: ({agent_pos[0]}, {agent_pos[1]})", transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left', fontsize=10, fontweight='bold')
    
    return fig

def display_algorithm_comparison():
    """显示算法对比说明"""
    st.subheader("Q-Learning 与 SARSA 算法核心区别")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Q-Learning (离策略)**")
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
        <strong>Q值更新公式:</strong><br>
        <span style="color: red; font-weight: bold;">Q(s,a) = Q(s,a) + α[r + γmax(Q(s',a')) - Q(s,a)]</span><br><br>
        • 基于贪心策略更新<br>
        • 假设下一步采取最优动作<br>
        • 更激进，可能更危险
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**SARSA (在线策略)**")
        st.markdown("""
        <div style="background-color: #fff8f0; padding: 10px; border-radius: 5px;">
        <strong>Q值更新公式:</strong><br>
        <span style="color: blue; font-weight: bold;">Q(s,a) = Q(s,a) + α[r + γQ(s',a') - Q(s,a)]</span><br><br>
        • 基于实际执行动作更新<br>
        • 考虑实际下一步动作<br>
        • 更保守，考虑策略
        </div>
        """, unsafe_allow_html=True)

def main():
    # 初始化参数
    grid_size = 5
    start_pos = (grid_size - 1, 0)  # 左下角
    goal_pos = (0, grid_size - 1)   # 右上角
    actions = ['up', 'down', 'left', 'right']
    
    # 初始化Q表
    q_table_q_learning = initialize_q_table(grid_size, grid_size, actions)
    q_table_sarsa = initialize_q_table(grid_size, grid_size, actions)
    
    # 显示算法对比说明
    display_algorithm_comparison()
    
    # 创建占位符用于动态更新
    plot_placeholder_q = st.empty()
    plot_placeholder_s = st.empty()
    status_placeholder = st.empty()
    
    # 添加进度条
    progress_bar = st.progress(0)
    
    # 添加统计信息显示
    stats_placeholder = st.empty()
    
    # 添加控制按钮
    col1, col2 = st.columns(2)
    start_button = col1.button("开始训练")
    reset_button = col2.button("重置")
    
    # 添加动画速度控制
    animation_speed = st.sidebar.slider("动画速度", 0.01, 1.0, 0.1, 0.01)
    
    if start_button:
        # 开始训练
        total_steps = num_episodes * 50
        current_step = 0
        successful_episodes_q = 0
        successful_episodes_s = 0
        
        # 显示初始状态
        fig_q = visualize_grid_world(grid_size, start_pos, goal_pos, q_table_q_learning, 0, 0, "Q-Learning")
        fig_s = visualize_grid_world(grid_size, start_pos, goal_pos, q_table_sarsa, 0, 0, "SARSA")
        plot_placeholder_q.pyplot(fig_q)
        plot_placeholder_s.pyplot(fig_s)
        
        for episode in range(num_episodes):
            # 重置智能体位置
            current_state_q = start_pos
            current_state_s = start_pos
            
            # 为SARSA获取第一个动作
            current_action_s = get_action(current_state_s, q_table_sarsa, exploration_rate, actions)
            
            # 显示当前轮次信息
            status_placeholder.write(f"正在执行第 {episode + 1} 轮训练...")
            
            # 每一轮的步骤
            for step in range(50):  # 限制每轮最大步数
                # Q-Learning部分
                # 选择动作
                action_q = get_action(current_state_q, q_table_q_learning, exploration_rate, actions)
                
                # 移动智能体
                next_state_q = move_agent(current_state_q, action_q, grid_size, grid_size)
                
                # 计算奖励
                reward_q = calculate_reward(current_state_q, goal_pos, next_state_q)
                
                # 更新Q值
                q_table_q_learning = update_q_value_q_learning(
                    q_table_q_learning, current_state_q, action_q, reward_q, next_state_q, learning_rate, discount_factor
                )
                
                # SARSA部分
                # 移动智能体
                next_state_s = move_agent(current_state_s, current_action_s, grid_size, grid_size)
                
                # 计算奖励
                reward_s = calculate_reward(current_state_s, goal_pos, next_state_s)
                
                # 选择下一个动作
                next_action_s = get_action(next_state_s, q_table_sarsa, exploration_rate, actions)
                
                # 更新Q值
                q_table_sarsa = update_q_value_sarsa(
                    q_table_sarsa, current_state_s, current_action_s, reward_s, next_state_s, next_action_s, learning_rate, discount_factor
                )
                
                # 可视化当前状态
                fig_q = visualize_grid_world(grid_size, next_state_q, goal_pos, q_table_q_learning, episode, step, "Q-Learning")
                fig_s = visualize_grid_world(grid_size, next_state_s, goal_pos, q_table_sarsa, episode, step, "SARSA")
                plot_placeholder_q.pyplot(fig_q)
                plot_placeholder_s.pyplot(fig_s)
                
                # 如果到达目标，结束当前轮次
                if is_goal_reached(next_state_q, goal_pos):
                    successful_episodes_q += 1
                    
                if is_goal_reached(next_state_s, goal_pos):
                    successful_episodes_s += 1
                    break  # 结束这一轮
                
                # 更新当前状态
                current_state_q = next_state_q
                current_state_s = next_state_s
                current_action_s = next_action_s
                
                # 更新进度条
                current_step += 1
                progress_bar.progress(min(current_step / total_steps, 1.0))
                
                # 更新统计信息
                success_rate_q = successful_episodes_q / (episode + 1) * 100 if episode > 0 else 0
                success_rate_s = successful_episodes_s / (episode + 1) * 100 if episode > 0 else 0
                stats_placeholder.markdown(f"**Q-Learning 成功率**: {success_rate_q:.1f}% | **SARSA 成功率**: {success_rate_s:.1f}% | **Q-Learning 成功轮次**: {successful_episodes_q}/{episode + 1} | **SARSA 成功轮次**: {successful_episodes_s}/{episode + 1}")
                
                # 添加短暂延迟以创建动画效果
                time.sleep(animation_speed)
        
        # 训练完成
        status_placeholder.write("训练完成！")
        st.success("Q-Learning和SARSA算法训练完成，智能体已学习到最优路径！")
        
        # 显示最终Q表对比
        st.subheader("最终Q表对比")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Q-Learning 最终Q表**")
            for i in range(grid_size):
                for j in range(grid_size):
                    if (i, j) != goal_pos:
                        q_values = q_table_q_learning[i, j]
                        st.write(f"位置({i},{j}): U={q_values[0]:.2f}, D={q_values[1]:.2f}, L={q_values[2]:.2f}, R={q_values[3]:.2f}")
        
        with col2:
            st.markdown("**SARSA 最终Q表**")
            for i in range(grid_size):
                for j in range(grid_size):
                    if (i, j) != goal_pos:
                        q_values = q_table_sarsa[i, j]
                        st.write(f"位置({i},{j}): U={q_values[0]:.2f}, D={q_values[1]:.2f}, L={q_values[2]:.2f}, R={q_values[3]:.2f}")

    if reset_button:
        # 重新初始化Q表和状态
        q_table_q_learning = initialize_q_table(grid_size, grid_size, actions)
        q_table_sarsa = initialize_q_table(grid_size, grid_size, actions)
        
        # 显示重置后的初始状态
        fig_q = visualize_grid_world(grid_size, start_pos, goal_pos, q_table_q_learning, 0, 0, "Q-Learning")
        fig_s = visualize_grid_world(grid_size, start_pos, goal_pos, q_table_sarsa, 0, 0, "SARSA")
        plot_placeholder_q.pyplot(fig_q)
        plot_placeholder_s.pyplot(fig_s)
        
        status_placeholder.write("已重置Q表，准备开始新的训练。")
        stats_placeholder.empty()
        progress_bar.progress(0)

    # 添加显示最终Q表的按钮
    if st.button("显示最终Q表对比"):
        st.subheader("最终Q表对比")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Q-Learning 最终Q表**")
            for i in range(grid_size):
                for j in range(grid_size):
                    if (i, j) != goal_pos:
                        q_values = q_table_q_learning[i, j]
                        st.write(f"位置({i},{j}): U={q_values[0]:.2f}, D={q_values[1]:.2f}, L={q_values[2]:.2f}, R={q_values[3]:.2f}")
        
        with col2:
            st.markdown("**SARSA 最终Q表**")
            for i in range(grid_size):
                for j in range(grid_size):
                    if (i, j) != goal_pos:
                        q_values = q_table_sarsa[i, j]
                        st.write(f"位置({i},{j}): U={q_values[0]:.2f}, D={q_values[1]:.2f}, L={q_values[2]:.2f}, R={q_values[3]:.2f}")

if __name__ == "__main__":
    main()