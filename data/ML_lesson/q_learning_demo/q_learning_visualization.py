import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']  # 使用英文字体以避免显示问题
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置页面标题
st.title("Q-Learning vs SARSA vs TD(λ) Algorithm Comparison Visualization Demo")

# 创建侧边栏用于参数调整
st.sidebar.header("Global Settings")
num_episodes = st.sidebar.slider("Training Episodes", 10, 100, 30, 5)

st.sidebar.header("Algorithm-Specific Parameters")
st.sidebar.subheader("Q-Learning & SARSA Parameters")
learning_rate = st.sidebar.slider("Learning Rate (α)", 0.1, 1.0, 0.5, 0.1)
exploration_rate = st.sidebar.slider("Exploration Rate (ε)", 0.0, 1.0, 0.3, 0.1)

st.sidebar.subheader("Common Discount Factor")
discount_factor = st.sidebar.slider("Discount Factor (γ)", 0.0, 1.0, 0.9, 0.1)

st.sidebar.subheader("TD(λ) Specific Parameter")
lambda_param = st.sidebar.slider("Lambda Parameter (λ)", 0.0, 1.0, 0.8, 0.1)  # TD(lambda) 参数

# 添加参数说明
with st.sidebar.expander("Parameter Explanations", expanded=False):
    st.markdown("""
    **Learning Rate (α)**: Controls how fast new information replaces old information
    **Exploration Rate (ε)**: Determines balance between exploration and exploitation
    **Discount Factor (γ)**: Weighs current rewards versus future rewards importance
    **Lambda Parameter (λ)**: Balances immediate and long-term impact, MC method at 0, DP method at 1
    **Training Episodes**: Number of times agent traverses environment
    """)

def initialize_q_table(rows, cols, actions):
    """Initialize Q-table"""
    return np.zeros((rows, cols, len(actions)))

def get_action(state, q_table, exploration_rate, actions):
    """Choose action based on ε-greedy policy"""
    if random.random() < exploration_rate:
        return random.choice(actions)
    else:
        row, col = state
        return actions[np.argmax(q_table[row, col])]

def calculate_reward(state, goal, next_state, obstacle_positions=None):
    """Calculate reward function"""
    if next_state == goal:
        return 100  # High reward for reaching the goal
    elif state == next_state:
        return -10  # Penalty for hitting a wall
    else:
        # Use Manhattan distance as part of the reward, closer to the goal gives higher reward
        distance_to_goal = abs(next_state[0] - goal[0]) + abs(next_state[1] - goal[1])
        return -distance_to_goal  # Larger distance gives larger penalty

def update_q_value_q_learning(q_table, state, action, reward, next_state, goal, learning_rate, discount_factor=0.9, actions=None):
    """Update Q-value - Q-Learning algorithm"""
    row, col = state
    next_row, next_col = next_state
    if actions is None:
        actions = ['up', 'down', 'left', 'right']
    action_idx = actions.index(action)
    
    # Q-learning formula: Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
    current_q = q_table[row, col, action_idx]
    max_next_q = np.max(q_table[next_row, next_col]) if next_state != goal else 0
    new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
    q_table[row, col, action_idx] = new_q
    
    return q_table

def update_q_value_sarsa(q_table, state, action, reward, next_state, next_action, goal, learning_rate, discount_factor=0.9, actions=None):
    """Update Q-value - SARSA algorithm"""
    row, col = state
    next_row, next_col = next_state
    if actions is None:
        actions = ['up', 'down', 'left', 'right']
    action_idx = actions.index(action)
    next_action_idx = actions.index(next_action) if next_action is not None else 0
    
    # SARSA formula: Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
    # Note: Here we use the actual next action next_action, not the action with the maximum Q-value
    current_q = q_table[row, col, action_idx]
    next_q = q_table[next_row, next_col, next_action_idx] if next_state != goal else 0
    new_q = current_q + learning_rate * (reward + discount_factor * next_q - current_q)
    q_table[row, col, action_idx] = new_q
    
    return q_table

def update_q_value_td_lambda(q_table, eligibility_trace, state, action, reward, next_state, next_action, goal, learning_rate, discount_factor, lambda_param, actions=None):
    """Update Q-value - TD(λ) algorithm (using eligibility trace)"""
    row, col = state
    next_row, next_col = next_state
    if actions is None:
        actions = ['up', 'down', 'left', 'right']
    action_idx = actions.index(action)
    
    # Calculate TD error
    if next_state != goal:
        next_action_idx = actions.index(next_action) if next_action is not None else 0
        td_error = reward + discount_factor * q_table[next_row, next_col, next_action_idx] - q_table[row, col, action_idx]
    else:
        td_error = reward - q_table[row, col, action_idx]
    
    # Update eligibility trace
    eligibility_trace[row, col, action_idx] += 1
    
    # Update Q-values for all state-action pairs
    q_table += learning_rate * td_error * eligibility_trace
    
    # Decay eligibility trace
    eligibility_trace *= discount_factor * lambda_param
    
    return q_table, eligibility_trace

def move_agent(state, action, rows, cols):
    """Move agent based on action"""
    row, col = state
    if action == 'up' and row > 0:
        return (row - 1, col)
    elif action == 'down' and row < rows - 1:
        return (row + 1, col)
    elif action == 'left' and col > 0:
        return (row, col - 1)
    elif action == 'right' and col < cols - 1:
        return (row, col + 1)
    return state  # Stay in the same position if unable to move

def is_goal_reached(state, goal):
    """Check if goal is reached"""
    return state == goal

def visualize_grid_world(grid_size, agent_pos, goal_pos, q_table, episode, step, algorithm_name, 
                        updated_sa_pairs=None, used_sa_pairs=None, extra_info=None, formula_display=None):
    """Visualize grid world"""
    fig, ax = plt.subplots(figsize=(16, 14))  # Increase figure size further to prevent overlap
    
    # Create color map
    cmap = ListedColormap(['white', 'lightblue', 'green'])
    
    # Create grid
    grid = np.zeros((grid_size, grid_size))
    
    # Mark goal position
    grid[goal_pos[0], goal_pos[1]] = 2
    
    # Mark agent position
    grid[agent_pos[0], agent_pos[1]] = 1
    
    # Plot grid
    ax.imshow(grid, cmap=cmap, interpolation='nearest')
    
    # Add axis labels
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels(range(grid_size))
    ax.set_yticklabels(range(grid_size))
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    
    # Add title, avoid overlapping with other information
    ax.set_title(f"{algorithm_name} Algorithm - Episode: {episode+1}, Step: {step+1}", fontsize=18, fontweight='bold', pad=60)
    
    # Display formula above the grid
    if formula_display:
        ax.text(0.5, 1.15, formula_display, transform=ax.transAxes, ha='center', va='center', 
                fontsize=13, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    # Display extra information above the grid
    if extra_info:
        info_text = ""
        for key, value in extra_info.items():
            info_text += f"{key}: {value}  "
        ax.text(0.5, 1.08, info_text, transform=ax.transAxes, ha='center', va='center', 
                fontsize=12, bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
    
    # Display Q-values on each cell - using a more concise way
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) != goal_pos:
                q_values = q_table[i, j]
                
                # Check if this position has any S-A pairs updated or used
                cell_updated = False
                cell_used = False
                
                if updated_sa_pairs:
                    for sa_pair in updated_sa_pairs:
                        if sa_pair[0] == i and sa_pair[1] == j:
                            cell_updated = True
                            break
                
                if used_sa_pairs:
                    for sa_pair in used_sa_pairs:
                        if sa_pair[0] == i and sa_pair[1] == j:
                            cell_used = True
                            break
                
                # Determine cell background color
                bg_color = 'white'
                if cell_updated:
                    bg_color = 'red'
                elif cell_used:
                    bg_color = 'orange'
                
                # Display Q-values, format more concisely, use English labels
                q_text = f"U:{q_values[0]:.1f}\nD:{q_values[1]:.1f}\nL:{q_values[2]:.1f}\nR:{q_values[3]:.1f}"
                
                # Adjust text style based on whether it has been updated or used
                text_color = 'black'
                if cell_updated:
                    text_color = 'white'
                
                ax.text(j, i, q_text, ha="center", va="center", 
                        fontsize=10, fontweight='bold', 
                        color=text_color, 
                        bbox=dict(boxstyle="round,pad=0.3", 
                                 facecolor=bg_color, alpha=0.8))
    
    # Display agent position
    ax.plot(agent_pos[1], agent_pos[0], 'ro', markersize=30, label='Agent', zorder=5)
    ax.plot(goal_pos[1], goal_pos[0], 'go', markersize=30, label='Goal', zorder=5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.15), fontsize=14)
    
    # Add description text
    ax.text(0.02, 0.98, f"Agent Position: ({agent_pos[0]},{agent_pos[1]})", transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left', fontsize=12, fontweight='bold')
    
    # Display updated and used Q-values, along with current state and reward, below the grid - in multiple lines (vertical layout)
    display_text_parts = []
    
    # Add current state information
    display_text_parts.append(f"Current State: S({agent_pos[0]},{agent_pos[1]})")
    
    # Add updated Q-values
    if updated_sa_pairs:
        for r, c, a_idx in updated_sa_pairs:
            action_name = ['Up', 'Down', 'Left', 'Right'][a_idx]
            q_val = q_table[r, c, a_idx]
            display_text_parts.append(f"Updated: S({r},{c})-{action_name}: {q_val:.2f}")
    
    # Add used Q-values
    if used_sa_pairs:
        filtered_used_pairs = [sa_pair for sa_pair in used_sa_pairs if not updated_sa_pairs or not any(sa_pair == up_pair for up_pair in updated_sa_pairs)]
        for r, c, a_idx in filtered_used_pairs:
            action_name = ['Up', 'Down', 'Left', 'Right'][a_idx]
            q_val = q_table[r, c, a_idx]
            display_text_parts.append(f"Used: S({r},{c})-{action_name}: {q_val:.2f}")
    
    # Add reward information if available in extra_info
    if extra_info and 'Reward' in extra_info:
        display_text_parts.append(f"Reward: {extra_info['Reward']}")
    
    # Add other parameters if available
    if extra_info:
        for key, value in extra_info.items():
            if key != 'Reward':  # Skip reward since we already displayed it separately
                display_text_parts.append(f"{key}: {value}")
    
    # Display all information at the bottom of the figure
    if display_text_parts:
        full_display_text = "\n".join(display_text_parts)
        
        # Display all information at the bottom of the figure
        ax.text(0.5, -0.25, full_display_text, transform=ax.transAxes, ha='center', va='center', 
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # Display color legend below the grid
    ax.text(0.5, -0.42, "Red=Updated SA-pairs, Orange=Used SA-pairs", transform=ax.transAxes, 
            ha='center', va='center', fontsize=11, 
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    return fig

def display_algorithm_comparison():
    """Display algorithm comparison information"""
    st.subheader("Q-Learning vs SARSA vs TD(λ) Algorithm Core Differences - Timing State Update Mechanism")
    
    st.markdown("""
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 8px; border-left: 5px solid #3366cc;">
    <h4>Key timing state update differences:</h4>
    <ul>
      <li><strong>Q-Learning (Off-policy)</strong>: Updates based on future optimal state, doesn't consider actual policy; update occurs after visiting state according to best expectation</li>
      <li><strong>SARSA (On-policy)</strong>: Updates based on actual executed action, considers current policy; update occurs after taking next actual action</li>
      <li><strong>TD(λ) (Eligibility Traces)</strong>: Combines historical trajectory weighting updates, balances immediate and long-term impact; update based on cumulative impact of all visited states</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Add algorithm steps explanation at the top
    st.markdown("""
    <div style="background-color: #e8f4fd; padding: 15px; border-radius: 8px; border-left: 5px solid #1f77b4; margin-top: 15px;">
    <h4>Algorithm Update Steps:</h4>
    <strong>Q-Learning Update Process:</strong>
    <ol>
      <li><strong>Choose action</strong>: Select action based on ε-greedy policy</li>
      <li><strong>Take action</strong>: Execute chosen action and observe next state and reward</li>
      <li><strong>Update Q-value</strong>: Update using max of next state's Q-values (greedy approach)</li>
      <li><strong>Move to next state</strong>: Transition to the observed next state</li>
    </ol>
    <strong>SARSA Update Process:</strong>
    <ol>
      <li><strong>Choose action</strong>: Select action based on current policy</li>
      <li><strong>Take action</strong>: Execute chosen action and observe next state and reward</li>
      <li><strong>Choose next action</strong>: Select next action based on current policy</li>
      <li><strong>Update Q-value</strong>: Update using next action's Q-value (actual policy execution)</li>
      <li><strong>Move to next state</strong>: Transition to the observed next state and next action</li>
    </ol>
    <strong>TD(λ) Update Process:</strong>
    <ol>
      <li><strong>Choose action</strong>: Select action based on current policy</li>
      <li><strong>Take action</strong>: Execute chosen action and observe next state and reward</li>
      <li><strong>Update eligibility trace</strong>: Increase eligibility trace for current state-action pair</li>
      <li><strong>Calculate TD error</strong>: Compute difference between expected and actual reward</li>
      <li><strong>Update Q-values</strong>: Update all state-action pairs based on eligibility trace</li>
      <li><strong>Decay eligibility trace</strong>: Reduce eligibility trace values to prepare for next step</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Q-Learning (Off-policy)**")
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px;">
        <strong>Q-value update formula:</strong><br>
        <span style="color: red; font-weight: bold;">Q(s<sub>t</sub>,a<sub>t</sub>) = Q(s<sub>t</sub>,a<sub>t</sub>) + α[r<sub>t+1</sub> + γmax(Q(s<sub>t+1</sub>,a')) - Q(s<sub>t</sub>,a<sub>t</sub>)]</span><br><br>
        • <strong>Timing update characteristic</strong>: Immediate update based on future optimal action<br>
        • Based on greedy strategy update<br>
        • Assumes next step takes optimal action<br>
        • More aggressive, possibly riskier
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**SARSA (On-policy)**")
        st.markdown("""
        <div style="background-color: #fff8f0; padding: 10px; border-radius: 5px;">
        <strong>Q-value update formula:</strong><br>
        <span style="color: blue; font-weight: bold;">Q(s<sub>t</sub>,a<sub>t</sub>) = Q(s<sub>t</sub>,a<sub>t</sub>) + α[r<sub>t+1</sub> + γQ(s<sub>t+1</sub>,a<sub>t+1</sub>) - Q(s<sub>t</sub>,a<sub>t</sub>)]</span><br><br>
        • <strong>Timing update characteristic</strong>: Delayed update based on actual executed action<br>
        • Based on actual executed action update<br>
        • Considers actual next action<br>
        • More conservative, considers policy
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("**TD(λ) (Eligibility Traces)**")
        st.markdown("""
        <div style="background-color: #f0fff0; padding: 10px; border-radius: 5px;">
        <strong>Q-value update formula:</strong><br>
        <span style="color: green; font-weight: bold;">Q(s,a) = Q(s,a) + αδe(s,a)</span><br><br>
        • <strong>Timing update characteristic</strong>: Cumulative update based on historical trajectory weights<br>
        • Uses eligibility traces to track history<br>
        • λ balances short-term and long-term impact<br>
        • Consides entire trajectory importance
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #fff2f2; padding: 15px; border-radius: 8px; border-left: 5px solid #ff4444;">
    <h4>Teaching Key Points:</h4>
    <ul>
      <li><strong>Q-Learning</strong>: Update timing occurs immediately after evaluating next possible best action</li>
      <li><strong>SARSA</strong>: Update timing occurs after actually executing next action</li>
      <li><strong>TD(λ)</strong>: Update timing occurs after entire sequence, according to trajectory importance for batch update</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def update_q_value_q_learning_with_display(q_table, state, action, reward, next_state, goal, learning_rate, discount_factor=0.9, actions=None):
    """Update Q-value - Q-Learning algorithm (with display)"""
    row, col = state
    next_row, next_col = next_state
    if actions is None:
        actions = ['up', 'down', 'left', 'right']
    action_idx = actions.index(action)
    
    # Get value before update
    current_q = q_table[row, col, action_idx]
    max_next_q = np.max(q_table[next_row, next_col]) if next_state != goal else 0
    
    # Calculate new value
    new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
    q_table[row, col, action_idx] = new_q
    
    # Create formula display text
    formula = f"Q(s,a)+α[r+γmax(Q(s',a'))-Q(s,a)]={current_q:.2f}+{learning_rate:.2f}[{reward}+{discount_factor:.2f}*{max_next_q:.2f}-{current_q:.2f}]={new_q:.2f}"
    
    # Return updated S-A pairs, used S-A pairs, and extra information
    updated_sa_pairs = [(row, col, action_idx)]  # Updated S-A pairs (row, col, action index)
    used_sa_pairs = [(next_row, next_col, np.argmax(q_table[next_row, next_col]))]  # Used S-A pairs (excluding the updated one)
    extra_info = {'Reward': f'{reward}', 'α': f'{learning_rate:.2f}', 'γ': f'{discount_factor:.2f}'}
    
    return q_table, updated_sa_pairs, used_sa_pairs, extra_info, formula

def update_q_value_sarsa_with_display(q_table, state, action, reward, next_state, next_action, goal, learning_rate, discount_factor=0.9, actions=None):
    """Update Q-value - SARSA algorithm (with display)"""
    row, col = state
    next_row, next_col = next_state
    if actions is None:
        actions = ['up', 'down', 'left', 'right']
    action_idx = actions.index(action)
    next_action_idx = actions.index(next_action) if next_action is not None else 0
    
    # Get value before update
    current_q = q_table[row, col, action_idx]
    next_q = q_table[next_row, next_col, next_action_idx] if next_state != goal else 0
    
    # Calculate new value
    new_q = current_q + learning_rate * (reward + discount_factor * next_q - current_q)
    q_table[row, col, action_idx] = new_q
    
    # Create formula display text
    formula = f"Q(s,a)+α[r+γQ(s',a')-Q(s,a)]={current_q:.2f}+{learning_rate:.2f}[{reward}+{discount_factor:.2f}*{next_q:.2f}-{current_q:.2f}]={new_q:.2f}"
    
    # Return updated S-A pairs, used S-A pairs, and extra information
    updated_sa_pairs = [(row, col, action_idx)]  # Updated S-A pairs
    used_sa_pairs = [(next_row, next_col, next_action_idx)]  # Used S-A pairs (excluding the updated one)
    extra_info = {'Reward': f'{reward}', 'α': f'{learning_rate:.2f}', 'γ': f'{discount_factor:.2f}'}
    
    return q_table, updated_sa_pairs, used_sa_pairs, extra_info, formula

def update_q_value_td_lambda_with_display(q_table, eligibility_trace, state, action, reward, next_state, next_action, goal, learning_rate, discount_factor, lambda_param, actions=None):
    """Update Q-value - TD(λ) algorithm (using eligibility trace, with display)"""
    row, col = state
    next_row, next_col = next_state
    if actions is None:
        actions = ['up', 'down', 'left', 'right']
    action_idx = actions.index(action)
    
    # Calculate TD error
    if next_state != goal:
        next_action_idx = actions.index(next_action) if next_action is not None else 0
        td_error = reward + discount_factor * q_table[next_row, next_col, next_action_idx] - q_table[row, col, action_idx]
    else:
        td_error = reward - q_table[row, col, action_idx]
    
    # Update eligibility trace - increase eligibility trace of the current state-action pair by 1
    eligibility_trace[row, col, action_idx] += 1
    
    # Get value before update
    old_q = q_table[row, col, action_idx]
    
    # Update Q-values for all state-action pairs - this is the core characteristic of TD(λ)
    q_table += learning_rate * td_error * eligibility_trace
    
    # Decay eligibility trace
    eligibility_trace *= discount_factor * lambda_param
    
    # Create formula display text
    new_q = q_table[row, col, action_idx]
    formula = f"Q(s,a)+αδe(s,a)={old_q:.2f}+{learning_rate:.2f}*{td_error:.2f}*{eligibility_trace[row, col, action_idx]/(discount_factor * lambda_param):.2f}={new_q:.2f}"
    
    # Determine all important updated S-A pairs
    updated_sa_pairs = []
    used_sa_pairs = [(next_row, next_col, next_action_idx if next_action is not None else 0)]  # Used state-action pairs (excluding the updated one)
    
    # Check only the eligibility trace values that are above threshold
    if discount_factor * lambda_param > 0:  # Avoid division by zero
        threshold_elig = eligibility_trace / (discount_factor * lambda_param)
        # Find indices where threshold is exceeded
        idxs = np.where(threshold_elig > 0.01)
        for i, j, k in zip(idxs[0], idxs[1], idxs[2]):
            updated_sa_pairs.append((i, j, k))
    else:
        # If lambda is 0, only current state-action pair is updated
        updated_sa_pairs = [(row, col, action_idx)]
    
    # If no updated positions are found, return at least the currently updated position.
    if not updated_sa_pairs:
        updated_sa_pairs = [(row, col, action_idx)]
    
    # Extra information
    extra_info = {'Reward': f'{reward}', 'α': f'{learning_rate:.2f}', 'γ': f'{discount_factor:.2f}', 'λ': f'{lambda_param:.2f}', 'TD_Error': f'{td_error:.2f}'}
    
    return q_table, updated_sa_pairs, used_sa_pairs, extra_info, formula

def main():
    # Initialize parameters - changed to 3x3 grid
    grid_size = 3
    start_pos = (grid_size - 1, 0)  # Bottom left
    goal_pos = (0, grid_size - 1)   # Top right
    actions = ['up', 'down', 'left', 'right']
    
    # Initialize Q-tables
    q_table_q_learning = initialize_q_table(grid_size, grid_size, actions)
    q_table_sarsa = initialize_q_table(grid_size, grid_size, actions)
    q_table_td_lambda = initialize_q_table(grid_size, grid_size, actions)
    
    # Initialize eligibility trace for TD(λ)
    eligibility_trace = np.zeros((grid_size, grid_size, len(actions)))
    
    # Display algorithm comparison explanation
    display_algorithm_comparison()
    
    # Display teaching demo points
    st.subheader("Teaching Demonstration Points")
    st.markdown("""
    <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px; border-left: 4px solid #1890ff;">
    <strong>Demonstration suggestions:</strong><br>
    1. <em>Parameter setting phase</em>: First use higher learning rate and exploration rate, to facilitate observing learning process<br>
    2. <em>Algorithm comparison phase</em>: Observe different learning trajectories of three algorithms under same environment<br>
    3. <em>Timing update differences</em>: Note timing differences when various algorithms update Q values<br>
    4. <em>Convergence characteristics</em>: Compare convergence speed and stability of three algorithms<br>
    5. <em>Policy differences</em>: Observe behavior differences between Q-Learning (off-policy) and SARSA/TD(λ) (on-policy)
    </div>
    """, unsafe_allow_html=True)
    
    # Add information about the visualization
    st.markdown("""
    <div style="background-color: #f0f8e6; padding: 10px; border-radius: 5px; border-left: 4px solid #32cd32;">
    <strong>Visualization Guide:</strong><br>
    • <em>Red cells</em> = State-Action pairs that were updated in this step<br>
    • <em>Orange cells</em> = State-Action pairs that were used in the update calculation<br>
    • <em>Green cell</em> = Goal position<br>
    • <em>Red circle</em> = Agent position<br>
    • <em>Q-values</em> = U:Up, D:Down, L:Left, R:Right<br>
    • <em>Bottom text</em> = Shows the updated Q-values in this step
    </div>
    """, unsafe_allow_html=True)
    
    # Define algorithm steps for reference (not displayed in expander)
    q_learning_steps = [
        "1. Choose action based on ε-greedy policy",
        "2. Take action and observe next state and reward",
        "3. Update Q-value using max of next state's Q-values",
        "4. Move to next state"
    ]
    
    sarsa_steps = [
        "1. Choose action based on current policy",
        "2. Take action and observe next state and reward",
        "3. Choose next action based on current policy",
        "4. Update Q-value using next action's Q-value",
        "5. Move to next state and next action"
    ]
    
    td_lambda_steps = [
        "1. Choose action based on current policy",
        "2. Take action and observe next state and reward",
        "3. Update eligibility trace for current state-action pair",
        "4. Calculate TD error",
        "5. Update Q-values for all state-action pairs based on eligibility trace",
        "6. Decay eligibility trace"
    ]
    
    # Create placeholders for dynamic updates with better layout
    st.subheader("Algorithm Comparison Visualization")
    st.markdown("""
    <div style="background-color: #e8f4fd; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
    <strong>Visualization Guide:</strong> Each column shows a different reinforcement learning algorithm.
    Red cells indicate state-action pairs that were updated in the current step,
    orange cells indicate state-action pairs that were used in the update calculation.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Q-Learning")
        st.markdown("**Strategy:** Off-policy, greedy approach")
        plot_placeholder_q = st.empty()
    with col2:
        st.markdown("### SARSA")
        st.markdown("**Strategy:** On-policy, actual action execution")
        plot_placeholder_s = st.empty()
    with col3:
        st.markdown("### TD(λ)")
        st.markdown("**Strategy:** Eligibility traces, historical weighting")
        plot_placeholder_tdl = st.empty()
    
    status_placeholder = st.empty()
    
    # Add progress bar
    progress_bar = st.progress(0)
    
    # Add statistics display
    stats_placeholder = st.empty()
    
    # Add control buttons
    col1, col2 = st.columns(2)
    start_button = col1.button("Start Training")
    reset_button = col2.button("Reset")
    
    # Add animation speed control
    animation_speed = st.sidebar.slider("Animation Speed", 0.01, 1.0, 0.1, 0.01)
    
    if start_button:
        # Start training
        total_steps = num_episodes * 50
        current_step = 0
        successful_episodes_q = 0
        successful_episodes_s = 0
        successful_episodes_t = 0
        
        # Display initial state (with display)
        fig_q = visualize_grid_world(grid_size, start_pos, goal_pos, q_table_q_learning, 0, 0, "Q-Learning")
        fig_s = visualize_grid_world(grid_size, start_pos, goal_pos, q_table_sarsa, 0, 0, "SARSA")
        fig_t = visualize_grid_world(grid_size, start_pos, goal_pos, q_table_td_lambda, 0, 0, "TD(λ)")
        plot_placeholder_q.pyplot(fig_q)
        plot_placeholder_s.pyplot(fig_s)
        plot_placeholder_tdl.pyplot(fig_t)
        
        for episode in range(num_episodes):
            # Reset agent position
            current_state_q = start_pos
            current_state_s = start_pos
            current_state_t = start_pos
            
            # Get first action for SARSA
            current_action_s = get_action(current_state_s, q_table_sarsa, exploration_rate, actions)
            
            # Get first action for TD(λ)
            current_action_t = get_action(current_state_t, q_table_td_lambda, exploration_rate, actions)
            
            # Display current episode information
            status_placeholder.write(f"Training episode {episode + 1} in progress...")
            
            # Steps in each episode
            for step in range(50):  # Limit maximum steps per episode
                # Q-Learning part
                # Choose action
                action_q = get_action(current_state_q, q_table_q_learning, exploration_rate, actions)
                
                # Move agent
                next_state_q = move_agent(current_state_q, action_q, grid_size, grid_size)
                
                # Calculate reward
                reward_q = calculate_reward(current_state_q, goal_pos, next_state_q)
                
                # Update Q-value (with display)
                q_table_q_learning, updated_sa_pairs_q, used_sa_pairs_q, extra_info_q, formula_q = update_q_value_q_learning_with_display(
                    q_table_q_learning, current_state_q, action_q, reward_q, next_state_q, goal_pos, learning_rate, discount_factor
                )
                
                # SARSA part
                # Move agent
                next_state_s = move_agent(current_state_s, current_action_s, grid_size, grid_size)
                
                # Calculate reward
                reward_s = calculate_reward(current_state_s, goal_pos, next_state_s)
                
                # Choose next action
                next_action_s = get_action(next_state_s, q_table_sarsa, exploration_rate, actions)
                
                # Update Q-value (with display)
                q_table_sarsa, updated_sa_pairs_s, used_sa_pairs_s, extra_info_s, formula_s = update_q_value_sarsa_with_display(
                    q_table_sarsa, current_state_s, current_action_s, reward_s, next_state_s, next_action_s, goal_pos, learning_rate, discount_factor
                )
                
                # TD(λ) part
                # Move agent
                next_state_t = move_agent(current_state_t, current_action_t, grid_size, grid_size)
                
                # Calculate reward
                reward_t = calculate_reward(current_state_t, goal_pos, next_state_t)
                
                # Choose next action
                next_action_t = get_action(next_state_t, q_table_td_lambda, exploration_rate, actions)
                
                # Update Q-value (with display)
                if next_state_t != goal_pos:
                    q_table_td_lambda, updated_sa_pairs_t, used_sa_pairs_t, extra_info_t, formula_t = update_q_value_td_lambda_with_display(
                        q_table_td_lambda, eligibility_trace, current_state_t, current_action_t, 
                        reward_t, next_state_t, next_action_t, goal_pos, learning_rate, discount_factor, lambda_param
                    )
                else:
                    # We also need to return updated cell information when reaching the goal.
                    updated_sa_pairs_t = [current_state_t + (actions.index(current_action_t),)]  # Add action index
                    used_sa_pairs_t = [current_state_t + (actions.index(current_action_t),)]
                    extra_info_t = {'Status': 'Reached Goal'}
                    formula_t = "Goal reached, update stopped"
                
                            # Calculate current step for each algorithm (for highlighting)
                current_q_step = step % len(q_learning_steps)
                current_s_step = step % len(sarsa_steps)
                current_td_step = step % len(td_lambda_steps)
                
                # Update the plots in separate columns
                with col1:
                    fig_q = visualize_grid_world(grid_size, next_state_q, goal_pos, q_table_q_learning, episode, step, "Q-Learning", updated_sa_pairs_q, used_sa_pairs_q, extra_info_q, formula_q)
                    plot_placeholder_q.pyplot(fig_q)
                    # Display current step information for Q-Learning - show only current step and total steps
                    if 'q_step_placeholder' not in locals():
                        q_step_placeholder = st.empty()
                    q_step_placeholder.markdown(f"**Q-Learning Current Step**: {current_q_step + 1} of {len(q_learning_steps)}")
                    q_step_placeholder.markdown(f"**Current Step Description**: {q_learning_steps[current_q_step]}")
                
                with col2:
                    fig_s = visualize_grid_world(grid_size, next_state_s, goal_pos, q_table_sarsa, episode, step, "SARSA", updated_sa_pairs_s, used_sa_pairs_s, extra_info_s, formula_s)
                    plot_placeholder_s.pyplot(fig_s)
                    # Display current step information for SARSA - show only current step and total steps
                    if 's_step_placeholder' not in locals():
                        s_step_placeholder = st.empty()
                    s_step_placeholder.markdown(f"**SARSA Current Step**: {current_s_step + 1} of {len(sarsa_steps)}")
                    s_step_placeholder.markdown(f"**Current Step Description**: {sarsa_steps[current_s_step]}")
                
                with col3:
                    fig_t = visualize_grid_world(grid_size, next_state_t, goal_pos, q_table_td_lambda, episode, step, "TD(λ)", updated_sa_pairs_t, used_sa_pairs_t, extra_info_t, formula_t)
                    plot_placeholder_tdl.pyplot(fig_t)
                    # Display current step information for TD(λ) - show only current step and total steps
                    if 'td_step_placeholder' not in locals():
                        td_step_placeholder = st.empty()
                    td_step_placeholder.markdown(f"**TD(λ) Current Step**: {current_td_step + 1} of {len(td_lambda_steps)}")
                    td_step_placeholder.markdown(f"**Current Step Description**: {td_lambda_steps[current_td_step]}")
                
                # Removed timing sequence containers as they were deleted earlier
                
                # 如果到达目标，结束当前轮次
                if is_goal_reached(next_state_q, goal_pos):
                    successful_episodes_q += 1
                    break  # 结束这一轮
                
                if is_goal_reached(next_state_s, goal_pos):
                    successful_episodes_s += 1
                    break  # 结束这一轮
                
                if is_goal_reached(next_state_t, goal_pos):
                    successful_episodes_t += 1
                    break  # 结束这一轮
                
                # 更新当前状态
                current_state_q = next_state_q
                current_state_s = next_state_s
                current_state_t = next_state_t
                current_action_s = next_action_s
                current_action_t = next_action_t
                
                # 更新进度条
                current_step += 1
                progress_bar.progress(min(current_step / total_steps, 1.0))
                
                # 更新统计信息
                success_rate_q = successful_episodes_q / max(1, episode + 1) * 100
                success_rate_s = successful_episodes_s / max(1, episode + 1) * 100
                success_rate_t = successful_episodes_t / max(1, episode + 1) * 100
                stats_placeholder.markdown(f"**Q-Learning Success Rate**: {success_rate_q:.1f}% | **SARSA Success Rate**: {success_rate_s:.1f}% | **TD(λ) Success Rate**: {success_rate_t:.1f}% | **Q-Learning Successful Episodes**: {successful_episodes_q}/{episode + 1} | **SARSA Successful Episodes**: {successful_episodes_s}/{episode + 1} | **TD(λ) Successful Episodes**: {successful_episodes_t}/{episode + 1}")
                
                # Add short delay to create animation effect - slow down to show the process
                time.sleep(animation_speed * 2)  # Slow down by a factor of 2 to better show the update process.
        
        # Training complete
        status_placeholder.write("Training completed!")
        st.success("Q-Learning, SARSA and TD(λ) algorithm training completed, agent has learned optimal path!")
        
        # 显示最终Q表对比
        st.subheader("Final Q-Table Comparison")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Q-Learning Final Q-Table**")
            for i in range(grid_size):
                for j in range(grid_size):
                    if (i, j) != goal_pos:
                        q_values = q_table_q_learning[i, j]
                        st.write(f"Position({i},{j}): U={q_values[0]:.2f}, D={q_values[1]:.2f}, L={q_values[2]:.2f}, R={q_values[3]:.2f}")
        
        with col2:
            st.markdown("**SARSA Final Q-Table**")
            for i in range(grid_size):
                for j in range(grid_size):
                    if (i, j) != goal_pos:
                        q_values = q_table_sarsa[i, j]
                        st.write(f"Position({i},{j}): U={q_values[0]:.2f}, D={q_values[1]:.2f}, L={q_values[2]:.2f}, R={q_values[3]:.2f}")
        
        with col3:
            st.markdown("**TD(λ) Final Q-Table**")
            for i in range(grid_size):
                for j in range(grid_size):
                    if (i, j) != goal_pos:
                        q_values = q_table_td_lambda[i, j]
                        st.write(f"Position({i},{j}): U={q_values[0]:.2f}, D={q_values[1]:.2f}, L={q_values[2]:.2f}, R={q_values[3]:.2f}")

    if reset_button:
        # 重新初始化Q表和状态
        q_table_q_learning = initialize_q_table(grid_size, grid_size, actions)
        q_table_sarsa = initialize_q_table(grid_size, grid_size, actions)
        q_table_td_lambda = initialize_q_table(grid_size, grid_size, actions)
        eligibility_trace = np.zeros((grid_size, grid_size, len(actions)))  # 重置资格迹
        
        # 显示重置后的初始状态
        fig_q = visualize_grid_world(grid_size, start_pos, goal_pos, q_table_q_learning, 0, 0, "Q-Learning")
        fig_s = visualize_grid_world(grid_size, start_pos, goal_pos, q_table_sarsa, 0, 0, "SARSA")
        fig_t = visualize_grid_world(grid_size, start_pos, goal_pos, q_table_td_lambda, 0, 0, "TD(λ)")
        plot_placeholder_q.pyplot(fig_q)
        plot_placeholder_s.pyplot(fig_s)
        plot_placeholder_tdl.pyplot(fig_t)
        
        status_placeholder.write("Q-tables reset, ready for new training.")
        stats_placeholder.empty()
        progress_bar.progress(0)

    # 添加显示最终Q表的按钮
    if st.button("Show Final Q-Table Comparison"):
        st.subheader("Final Q-Table Comparison")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Q-Learning Final Q-Table**")
            for i in range(grid_size):
                for j in range(grid_size):
                    if (i, j) != goal_pos:
                        q_values = q_table_q_learning[i, j]
                        st.write(f"Position({i},{j}): U={q_values[0]:.2f}, D={q_values[1]:.2f}, L={q_values[2]:.2f}, R={q_values[3]:.2f}")
        
        with col2:
            st.markdown("**SARSA Final Q-Table**")
            for i in range(grid_size):
                for j in range(grid_size):
                    if (i, j) != goal_pos:
                        q_values = q_table_sarsa[i, j]
                        st.write(f"Position({i},{j}): U={q_values[0]:.2f}, D={q_values[1]:.2f}, L={q_values[2]:.2f}, R={q_values[3]:.2f}")
        
        with col3:
            st.markdown("**TD(λ) Final Q-Table**")
            for i in range(grid_size):
                for j in range(grid_size):
                    if (i, j) != goal_pos:
                        q_values = q_table_td_lambda[i, j]
                        st.write(f"Position({i},{j}): U={q_values[0]:.2f}, D={q_values[1]:.2f}, L={q_values[2]:.2f}, R={q_values[3]:.2f}")

if __name__ == "__main__":
    main()

