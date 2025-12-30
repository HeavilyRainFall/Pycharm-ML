"""
贝尔曼方程迭代演示程序
GUI界面展示强化学习中贝尔曼方程的迭代过程
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import threading

# 设置中文字体支持
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class BellmanEquationDemo:
    def __init__(self, root):
        self.root = root
        self.root.title("贝尔曼方程迭代演示")
        self.root.geometry("1000x700")
        
        # 初始化参数
        self.grid_size = 4
        self.gamma = 0.9  # 折扣因子
        self.threshold = 1e-4  # 收敛阈值（提高以减慢收敛）
        self.max_iterations = 100  # 最大迭代次数
        self.current_iteration = 0
        self.values = np.zeros((self.grid_size, self.grid_size))  # 状态价值函数
        self.rewards = np.zeros((self.grid_size, self.grid_size))  # 奖励矩阵
        self.policy = np.zeros((self.grid_size, self.grid_size), dtype=int)  # 策略 (0=上, 1=右, 2=下, 3=左)
        
        # 设置默认奖励和特殊状态
        self.terminal_pos = (0, 3)  # 终止状态
        self.hole_pos = (1, 1)  # 坑位
        self.start_pos = (3, 0)  # 起始状态
        
        # 设置奖励值
        self.rewards[self.terminal_pos] = 1.0  # 终止状态奖励
        self.rewards[self.hole_pos] = -1.0  # 坑位惩罚
        
        self.setup_ui()
        self.draw_initial_state()
    
    def setup_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 参数设置
        ttk.Label(control_frame, text="折扣因子 (γ):").pack(anchor=tk.W)
        self.gamma_var = tk.DoubleVar(value=self.gamma)
        gamma_scale = ttk.Scale(control_frame, from_=0.1, to=0.99, variable=self.gamma_var, 
                               orient=tk.HORIZONTAL, command=self.update_gamma)
        gamma_scale.pack(fill=tk.X, pady=5)
        self.gamma_label = ttk.Label(control_frame, text=f"{self.gamma:.2f}")
        self.gamma_label.pack()
        
        ttk.Label(control_frame, text="收敛阈值:").pack(anchor=tk.W, pady=(10, 0))
        self.threshold_var = tk.DoubleVar(value=self.threshold)
        threshold_entry = ttk.Entry(control_frame, textvariable=self.threshold_var)
        threshold_entry.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="最大迭代次数:").pack(anchor=tk.W)
        self.max_iter_var = tk.IntVar(value=self.max_iterations)
        max_iter_entry = ttk.Entry(control_frame, textvariable=self.max_iter_var)
        max_iter_entry.pack(fill=tk.X, pady=5)
        
        # 按钮
        ttk.Button(control_frame, text="开始迭代", command=self.start_iteration).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="单步迭代", command=self.step_iteration).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="重置", command=self.reset).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="可视化说明", command=self.visualize).pack(fill=tk.X, pady=5)
        
        # 迭代信息
        self.info_label = ttk.Label(control_frame, text=f"迭代次数: {self.current_iteration}")
        self.info_label.pack(pady=10)
        
        # 右侧显示区域
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建画布用于显示网格
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 数学公式显示
        formula_frame = ttk.LabelFrame(display_frame, text="贝尔曼方程", padding=10)
        formula_frame.pack(fill=tk.X, pady=(10, 0))
        
        formula_text = ("贝尔曼方程（状态价值函数）:\n"
                       "V(s) = R(s) + γ × max_a Σ p(s'|s,a) × V(s')\n\n"
                       "其中:\n"
                       "- V(s): 状态s的价值\n"
                       "- R(s): 在状态s的即时奖励\n"
                       "- γ: 折扣因子\n"
                       "- p(s'|s,a): 从状态s执行动作a转移到s'的概率")
        
        formula_label = ttk.Label(formula_frame, text=formula_text, justify=tk.LEFT)
        formula_label.pack()
    
    def update_gamma(self, value):
        self.gamma = float(value)
        self.gamma_label.config(text=f"{self.gamma:.2f}")
    
    def draw_initial_state(self):
        self.ax1.clear()
        self.ax1.set_title("状态价值函数")
        
        # 创建一个副本用于显示，对特殊状态使用不同颜色
        display_values = np.copy(self.values)
        
        # 标记特殊状态
        display_values[self.terminal_pos] = 2.0  # 终止状态设为高值（用红色表示）
        display_values[self.hole_pos] = -2.0    # 坑位设为低值（用蓝色表示）
        display_values[self.start_pos] = 1.5    # 起始状态设为中等正值（用黄色表示）
        
        im1 = self.ax1.imshow(display_values, cmap='RdYlGn', interpolation='nearest', vmin=-2, vmax=2)
        self.ax1.set_xticks(range(self.grid_size))
        self.ax1.set_yticks(range(self.grid_size))
        
        # 显示价值数值（特殊状态除外）
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) not in [self.terminal_pos, self.hole_pos, self.start_pos]:
                    self.ax1.text(j, i, f'{self.values[i, j]:.2f}', 
                                 ha='center', va='center', fontsize=8, color='black')
                elif (i, j) == self.terminal_pos:
                    self.ax1.text(j, i, 'T\n+1.0', 
                                 ha='center', va='center', fontsize=8, color='white',
                                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
                elif (i, j) == self.hole_pos:
                    self.ax1.text(j, i, 'H\n-1.0', 
                                 ha='center', va='center', fontsize=8, color='white',
                                 bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8))
                elif (i, j) == self.start_pos:
                    self.ax1.text(j, i, 'S\n0.0', 
                                 ha='center', va='center', fontsize=8, color='black',
                                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # 添加网格线
        for i in range(self.grid_size + 1):
            self.ax1.axhline(i - 0.5, color='black', linewidth=0.5)
            self.ax1.axvline(i - 0.5, color='black', linewidth=0.5)
        
        self.ax2.clear()
        self.ax2.set_title("奖励矩阵")
        im2 = self.ax2.imshow(self.rewards, cmap='RdBu', interpolation='nearest', vmin=-2, vmax=2)
        self.ax2.set_xticks(range(self.grid_size))
        self.ax2.set_yticks(range(self.grid_size))
        
        # 显示奖励数值
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) == self.terminal_pos:
                    self.ax2.text(j, i, 'T\n+1.0', 
                                 ha='center', va='center', fontsize=10, color='white', 
                                 bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
                elif (i, j) == self.hole_pos:
                    self.ax2.text(j, i, 'H\n-1.0', 
                                 ha='center', va='center', fontsize=10, color='white', 
                                 bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8))
                elif (i, j) == self.start_pos:
                    self.ax2.text(j, i, 'S\n0.0', 
                                 ha='center', va='center', fontsize=10, color='black',
                                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
                elif self.rewards[i, j] != 0 and (i, j) not in [self.terminal_pos, self.hole_pos]:
                    self.ax2.text(j, i, f'{self.rewards[i, j]:.1f}', 
                                 ha='center', va='center', fontsize=10, color='white', 
                                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        
        # 添加网格线
        for i in range(self.grid_size + 1):
            self.ax2.axhline(i - 0.5, color='black', linewidth=0.5)
            self.ax2.axvline(i - 0.5, color='black', linewidth=0.5)
        
        # 添加颜色条
        self.fig.colorbar(im1, ax=self.ax1, shrink=0.8)
        self.fig.colorbar(im2, ax=self.ax2, shrink=0.8)
        
        self.canvas.draw()
    
    def value_iteration_step(self):
        """执行一次值迭代"""
        new_values = np.copy(self.values)
        delta = 0
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # 跳过终止状态
                if (i, j) == self.terminal_pos:
                    continue
                
                # 计算四个方向的预期价值
                expected_values = []
                
                # 上 (0)
                if i > 0:
                    expected_values.append(self.rewards[i, j] + self.gamma * self.values[i-1, j])
                else:
                    expected_values.append(self.rewards[i, j] + self.gamma * self.values[i, j])
                
                # 右 (1)
                if j < self.grid_size - 1:
                    expected_values.append(self.rewards[i, j] + self.gamma * self.values[i, j+1])
                else:
                    expected_values.append(self.rewards[i, j] + self.gamma * self.values[i, j])
                
                # 下 (2)
                if i < self.grid_size - 1:
                    expected_values.append(self.rewards[i, j] + self.gamma * self.values[i+1, j])
                else:
                    expected_values.append(self.rewards[i, j] + self.gamma * self.values[i, j])
                
                # 左 (3)
                if j > 0:
                    expected_values.append(self.rewards[i, j] + self.gamma * self.values[i, j-1])
                else:
                    expected_values.append(self.rewards[i, j] + self.gamma * self.values[i, j])
                
                # 选择最大价值
                new_values[i, j] = max(expected_values)
                
                # 更新策略
                self.policy[i, j] = np.argmax(expected_values)
                
                # 计算变化量
                delta = max(delta, abs(new_values[i, j] - self.values[i, j]))
        
        self.values = new_values
        return delta
    
    def start_iteration(self):
        """开始自动迭代直到收敛"""
        self.gamma = self.gamma_var.get()
        self.threshold = self.threshold_var.get()
        self.max_iterations = self.max_iter_var.get()
        
        def run_iteration():
            for _ in range(self.max_iterations):
                delta = self.value_iteration_step()
                self.current_iteration += 1
                self.info_label.config(text=f"迭代次数: {self.current_iteration}")
                
                # 更新显示
                self.root.after(0, self.draw_initial_state)  # 在主线程中更新UI
                
                # 检查是否收敛
                if delta < self.threshold:
                    self.root.after(0, lambda: messagebox.showinfo("完成", f"值迭代收敛！\n迭代次数: {self.current_iteration}"))
                    break
                
                # 添加延迟以更好地观察迭代过程
                time.sleep(0.3)
        
        # 在新线程中运行迭代以避免阻塞UI
        iteration_thread = threading.Thread(target=run_iteration)
        iteration_thread.daemon = True
        iteration_thread.start()
    
    def step_iteration(self):
        """单步迭代"""
        self.gamma = self.gamma_var.get()
        delta = self.value_iteration_step()
        self.current_iteration += 1
        self.info_label.config(text=f"迭代次数: {self.current_iteration}")
        
        # 更新显示
        self.draw_initial_state()
        
        # 检查收敛
        if delta < self.threshold_var.get():
            messagebox.showinfo("提示", f"算法已收敛！\n总迭代次数: {self.current_iteration}")
    
    def reset(self):
        """重置程序"""
        self.current_iteration = 0
        self.values = np.zeros((self.grid_size, self.grid_size))
        self.rewards = np.zeros((self.grid_size, self.grid_size))
        self.policy = np.zeros((self.grid_size, self.grid_size), dtype=int)
        
        # 设置奖励值
        self.rewards[self.terminal_pos] = 1.0  # 终止状态奖励
        self.rewards[self.hole_pos] = -1.0  # 坑位惩罚
        
        self.gamma = self.gamma_var.get()
        self.threshold = self.threshold_var.get()
        self.max_iterations = self.max_iter_var.get()
        
        self.info_label.config(text=f"迭代次数: {self.current_iteration}")
        self.draw_initial_state()
    
    def visualize(self):
        """显示详细的数学过程"""
        # 创建新窗口显示数学过程
        viz_window = tk.Toplevel(self.root)
        viz_window.title("贝尔曼方程数学过程")
        viz_window.geometry("800x600")
        
        # 添加滚动文本框
        text_frame = ttk.Frame(viz_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 添加数学过程说明
        math_process = f"""贝尔曼方程详解:

1. 基本概念:
   - 状态价值函数 V(s): 从状态s开始，遵循策略π的期望回报
   - 贝尔曼方程描述了当前状态与后续状态价值之间的关系

2. 贝尔曼期望方程:
   V(s) = R(s) + γ × Σ p(s'|s,a) × π(a|s) × V(s')
   
   其中:
   - R(s): 在状态s获得的即时奖励
   - γ: 折扣因子 (0 ≤ γ < 1)
   - p(s'|s,a): 从状态s执行动作a转移到s'的概率
   - π(a|s): 在状态s选择动作a的概率

3. 贝尔曼最优方程:
   V*(s) = R(s) + γ × max_a Σ p(s'|s,a) × V*(s')
   
   最优策略对应最优价值函数

4. 值迭代算法:
   V(k+1)(s) = R(s) + γ × max_a Σ p(s'|s,a) × V(k)(s')
   
   从任意V_0开始，迭代直到收敛

5. 在本演示中:
   - 网格世界: {self.grid_size}×{self.grid_size} 网格
   - 动作: 上、右、下、左四个方向
   - 转移: 确定性转移 (概率为1)
   - 特殊状态:
     * 起始状态 S: 位于 {self.start_pos}，用黄色标记
     * 终止状态 T: 位于 {self.terminal_pos}，奖励为+1.0，用红色标记
     * 坑位 H: 位于 {self.hole_pos}，奖励为-1.0，用蓝色标记
   - 折扣因子 γ = {self.gamma:.2f}
   
6. 迭代过程:
   对每个非终止状态(s), 计算:
   V(s) ← max_a [R(s) + γ × V(s')]
   
   其中s'是执行动作a后到达的下一个状态

7. 收敛判断:
   当所有状态的价值函数变化量小于阈值时停止迭代:
   max|V(k+1)(s) - V(k)(s)| < 阈值"""
        
        text_widget.insert(tk.END, math_process)
        text_widget.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    app = BellmanEquationDemo(root)
    root.mainloop()


if __name__ == "__main__":
    main()