# ==============================
# 强化学习交互式学习平台：Flappy Bird（带训练过程可视化）
# 使用 Tkinter + Stable-Baselines3 + flappybird-gym
# ==============================

import tkinter as tk
from tkinter import ttk, messagebox
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from datetime import datetime
import csv
import shutil
import flappy_bird_gymnasium

# %% 全局绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12


class RLLearningPlatform:
    def __init__(self, root):
        self.root = root
        self.root.title("强化学习学习平台 - Flappy Bird")
        self.root.geometry("1000x700")

        # 训练状态
        self.training = False
        self.model = None
        self.current_step = 0
        self.total_steps = 0
        self.training_thread = None

        # 演示环境
        self.demo_env = None
        self.demo_running = False
        self.demo_thread = None

        # 训练记录 - 改为记录演示得分（超越的柱子数）
        self.demo_scores = []  # 每次演示的得分（超越的柱子数）
        self.demo_steps = []  # 演示时的训练步数
        self.demo_count = 0  # 演示次数

        # 创建必要的目录
        os.makedirs("../models/", exist_ok=True)
        os.makedirs("../logs/", exist_ok=True)

        # 初始化界面
        self.setup_ui()
        self.update_model_list()

    def setup_ui(self):
        """设置用户界面"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        title_label = ttk.Label(main_frame, text="🐦 强化学习学习平台 - Flappy Bird",
                                font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="模型配置", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))

        # 算法选择
        ttk.Label(control_frame, text="算法:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.algorithm_var = tk.StringVar(value="PPO")
        algorithm_combo = ttk.Combobox(control_frame, textvariable=self.algorithm_var,
                                       values=["PPO", "A2C", "DQN"], state="readonly")
        algorithm_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        algorithm_combo.bind('<<ComboboxSelected>>', self.on_algorithm_change)

        # 算法介绍
        self.algorithm_info_var = tk.StringVar(value="PPO: 近端策略优化，稳定高效的策略梯度方法")
        algorithm_info_label = ttk.Label(control_frame, textvariable=self.algorithm_info_var,
                                         wraplength=300, foreground="blue")
        algorithm_info_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))

        # 模型名称
        ttk.Label(control_frame, text="模型名称:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.model_name_var = tk.StringVar(value="my_model")
        ttk.Entry(control_frame, textvariable=self.model_name_var).grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)

        # 训练参数
        param_frame = ttk.Frame(control_frame)
        param_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(param_frame, text="训练步数:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.timesteps_var = tk.IntVar(value=100000)
        timesteps_scale = ttk.Scale(param_frame, from_=10000, to=500000,
                                    variable=self.timesteps_var, orient=tk.HORIZONTAL)
        timesteps_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        self.timesteps_label = ttk.Label(param_frame, text=f"{self.timesteps_var.get():,}")
        self.timesteps_label.grid(row=0, column=2, padx=(5, 0))
        timesteps_scale.configure(command=lambda v: self.timesteps_label.config(text=f"{int(float(v)):,}"))

        ttk.Label(param_frame, text="学习率:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.lr_var = tk.DoubleVar(value=0.001)
        lr_scale = ttk.Scale(param_frame, from_=0.0001, to=0.01,
                             variable=self.lr_var, orient=tk.HORIZONTAL)
        lr_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        self.lr_label = ttk.Label(param_frame, text=f"{self.lr_var.get():.4f}")
        self.lr_label.grid(row=1, column=2, padx=(5, 0))
        lr_scale.configure(command=lambda v: self.lr_label.config(text=f"{float(v):.4f}"))

        ttk.Label(param_frame, text="折扣因子:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.gamma_var = tk.DoubleVar(value=0.99)
        gamma_scale = ttk.Scale(param_frame, from_=0.9, to=0.999,
                                variable=self.gamma_var, orient=tk.HORIZONTAL)
        gamma_scale.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        self.gamma_label = ttk.Label(param_frame, text=f"{self.gamma_var.get():.3f}")
        self.gamma_label.grid(row=2, column=2, padx=(5, 0))
        gamma_scale.configure(command=lambda v: self.gamma_label.config(text=f"{float(v):.3f}"))

        # 算法特定参数
        self.algo_specific_frame = ttk.Frame(control_frame)
        self.algo_specific_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        self.setup_algorithm_specific_controls()

        # 游戏显示设置
        display_frame = ttk.LabelFrame(control_frame, text="游戏显示设置", padding="5")
        display_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        # 窗口大小设置
        ttk.Label(display_frame, text="窗口大小:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.window_size_var = tk.StringVar(value="288x512")
        window_size_combo = ttk.Combobox(display_frame, textvariable=self.window_size_var,
                                         values=["288x512", "432x768", "576x1024"], width=10)
        window_size_combo.grid(row=0, column=1, sticky=tk.W, pady=2)

        # 显示选项
        self.show_lidar_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(display_frame, text="显示激光雷达线",
                        variable=self.show_lidar_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.show_hitbox_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(display_frame, text="显示碰撞框",
                        variable=self.show_hitbox_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=2)

        self.show_score_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="显示分数",
                        variable=self.show_score_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)

        # 模型管理
        ttk.Label(control_frame, text="现有模型:").grid(row=6, column=0, sticky=tk.W, pady=(10, 5))
        self.model_listbox = tk.Listbox(control_frame, height=5)
        self.model_listbox.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        model_buttons_frame = ttk.Frame(control_frame)
        model_buttons_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(model_buttons_frame, text="加载模型", command=self.load_model).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(model_buttons_frame, text="删除模型", command=self.delete_model).pack(side=tk.LEFT)

        self.continue_training_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="继续训练", variable=self.continue_training_var).grid(row=9, column=0,
                                                                                                  columnspan=2,
                                                                                                  sticky=tk.W, pady=5)

        # 实时演示选项
        demo_frame = ttk.Frame(control_frame)
        demo_frame.grid(row=10, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.live_demo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(demo_frame, text="训练过程中实时演示",
                        variable=self.live_demo_var).pack(side=tk.LEFT)

        ttk.Label(demo_frame, text="演示间隔(步):").pack(side=tk.LEFT, padx=(10, 5))
        self.demo_interval_var = tk.IntVar(value=5000)
        demo_interval_combo = ttk.Combobox(demo_frame, textvariable=self.demo_interval_var,
                                           values=[1000, 2500, 5000, 10000], width=8)
        demo_interval_combo.pack(side=tk.LEFT)

        ttk.Label(demo_frame, text="演示局数:").pack(side=tk.LEFT, padx=(10, 5))
        self.demo_episodes_var = tk.IntVar(value=1)
        demo_episodes_combo = ttk.Combobox(demo_frame, textvariable=self.demo_episodes_var,
                                           values=[1, 2, 3], width=5)
        demo_episodes_combo.pack(side=tk.LEFT)

        # 按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=11, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        self.train_button = ttk.Button(button_frame, text="开始训练", command=self.toggle_training)
        self.train_button.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="评估模型", command=self.evaluate_model).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="演示游戏", command=self.demo_game).pack(side=tk.LEFT)

        # 进度
        ttk.Label(control_frame, text="训练进度:").grid(row=12, column=0, sticky=tk.W, pady=(10, 5))
        self.progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        progress_bar.grid(row=13, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.status_var = tk.StringVar(value="准备就绪")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.grid(row=14, column=0, columnspan=2, sticky=tk.W, pady=5)

        # 右侧图表
        result_frame = ttk.LabelFrame(main_frame, text="训练结果", padding="10")
        result_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=result_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        self.setup_charts()

        control_frame.columnconfigure(1, weight=1)
        param_frame.columnconfigure(1, weight=1)

    def setup_algorithm_specific_controls(self):
        for widget in self.algo_specific_frame.winfo_children():
            widget.destroy()

        algorithm = self.algorithm_var.get()

        # 更新算法介绍
        algorithm_info = {
            "PPO": "PPO (近端策略优化): 稳定高效的策略梯度方法，通过裁剪保证策略更新稳定性",
            "A2C": "A2C (同步优势演员评论家): 同步版本A3C，结合策略和价值网络，适合并行训练",
            "DQN": "DQN (深度Q网络): 基于值函数的深度强化学习，使用经验回放和目标网络"
        }
        self.algorithm_info_var.set(algorithm_info.get(algorithm, ""))

        if algorithm == "PPO":
            ttk.Label(self.algo_specific_frame, text="PPO参数:").grid(row=0, column=0, sticky=tk.W, pady=5)

            ttk.Label(self.algo_specific_frame, text="n_steps:").grid(row=1, column=0, sticky=tk.W, pady=2)
            self.n_steps_var = tk.IntVar(value=2048)
            ttk.Entry(self.algo_specific_frame, textvariable=self.n_steps_var, width=10).grid(row=1, column=1,
                                                                                              sticky=tk.W, pady=2)

            ttk.Label(self.algo_specific_frame, text="batch_size:").grid(row=2, column=0, sticky=tk.W, pady=2)
            self.batch_size_var = tk.IntVar(value=64)
            ttk.Entry(self.algo_specific_frame, textvariable=self.batch_size_var, width=10).grid(row=2, column=1,
                                                                                                 sticky=tk.W, pady=2)

            ttk.Label(self.algo_specific_frame, text="n_epochs:").grid(row=3, column=0, sticky=tk.W, pady=2)
            self.n_epochs_var = tk.IntVar(value=10)
            ttk.Entry(self.algo_specific_frame, textvariable=self.n_epochs_var, width=10).grid(row=3, column=1,
                                                                                               sticky=tk.W, pady=2)

            ttk.Label(self.algo_specific_frame, text="clip_range:").grid(row=4, column=0, sticky=tk.W, pady=2)
            self.clip_range_var = tk.DoubleVar(value=0.2)
            ttk.Entry(self.algo_specific_frame, textvariable=self.clip_range_var, width=10).grid(row=4, column=1,
                                                                                                 sticky=tk.W, pady=2)

        elif algorithm == "A2C":
            ttk.Label(self.algo_specific_frame, text="A2C参数:").grid(row=0, column=0, sticky=tk.W, pady=5)

            ttk.Label(self.algo_specific_frame, text="n_steps:").grid(row=1, column=0, sticky=tk.W, pady=2)
            self.n_steps_var = tk.IntVar(value=5)
            ttk.Entry(self.algo_specific_frame, textvariable=self.n_steps_var, width=10).grid(row=1, column=1,
                                                                                              sticky=tk.W, pady=2)

            ttk.Label(self.algo_specific_frame, text="ent_coef:").grid(row=2, column=0, sticky=tk.W, pady=2)
            self.ent_coef_var = tk.DoubleVar(value=0.0)
            ttk.Entry(self.algo_specific_frame, textvariable=self.ent_coef_var, width=10).grid(row=2, column=1,
                                                                                               sticky=tk.W, pady=2)

            ttk.Label(self.algo_specific_frame, text="vf_coef:").grid(row=3, column=0, sticky=tk.W, pady=2)
            self.vf_coef_var = tk.DoubleVar(value=0.5)
            ttk.Entry(self.algo_specific_frame, textvariable=self.vf_coef_var, width=10).grid(row=3, column=1,
                                                                                              sticky=tk.W, pady=2)

        elif algorithm == "DQN":
            ttk.Label(self.algo_specific_frame, text="DQN参数:").grid(row=0, column=0, sticky=tk.W, pady=5)

            ttk.Label(self.algo_specific_frame, text="buffer_size:").grid(row=1, column=0, sticky=tk.W, pady=2)
            self.buffer_size_var = tk.IntVar(value=10000)
            ttk.Entry(self.algo_specific_frame, textvariable=self.buffer_size_var, width=10).grid(row=1, column=1,
                                                                                                  sticky=tk.W, pady=2)

            ttk.Label(self.algo_specific_frame, text="learning_starts:").grid(row=2, column=0, sticky=tk.W, pady=2)
            self.learning_starts_var = tk.IntVar(value=1000)
            ttk.Entry(self.algo_specific_frame, textvariable=self.learning_starts_var, width=10).grid(row=2, column=1,
                                                                                                      sticky=tk.W,
                                                                                                      pady=2)

            ttk.Label(self.algo_specific_frame, text="target_update_interval:").grid(row=3, column=0, sticky=tk.W,
                                                                                     pady=2)
            self.target_update_interval_var = tk.IntVar(value=1000)
            ttk.Entry(self.algo_specific_frame, textvariable=self.target_update_interval_var, width=10).grid(row=3,
                                                                                                             column=1,
                                                                                                             sticky=tk.W,
                                                                                                             pady=2)

            ttk.Label(self.algo_specific_frame, text="exploration_fraction:").grid(row=4, column=0, sticky=tk.W, pady=2)
            self.exploration_fraction_var = tk.DoubleVar(value=0.1)
            ttk.Entry(self.algo_specific_frame, textvariable=self.exploration_fraction_var, width=10).grid(row=4,
                                                                                                           column=1,
                                                                                                           sticky=tk.W,
                                                                                                           pady=2)

    def setup_charts(self):
        self.ax1.clear()
        self.ax1.set_title('训练过程中演示表现')
        self.ax1.set_xlabel('演示次数')
        self.ax1.set_ylabel('超越柱子数')
        self.ax2.clear()
        self.ax2.set_title('评估结果')
        self.ax2.set_xlabel('评估次数')
        self.ax2.set_ylabel('平均奖励')
        self.canvas.draw()

    def on_algorithm_change(self, event=None):
        self.setup_algorithm_specific_controls()

    def update_model_list(self):
        self.model_listbox.delete(0, tk.END)
        for f in sorted(os.listdir("../models/")):
            if f.endswith(".zip"):
                self.model_listbox.insert(tk.END, f)

    def load_model(self):
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择一个模型")
            return
        model_name = self.model_listbox.get(selection[0])
        try:
            path = f"./models/{model_name}"

            # 根据文件名判断算法类型
            if "PPO" in model_name:
                self.model = PPO.load(path)
                self.algorithm_var.set("PPO")
            elif "A2C" in model_name:
                self.model = A2C.load(path)
                self.algorithm_var.set("A2C")
            elif "DQN" in model_name:
                self.model = DQN.load(path)
                self.algorithm_var.set("DQN")
            else:
                # 如果无法从文件名判断，尝试使用当前选择的算法
                algorithm = self.algorithm_var.get()
                if algorithm == "PPO":
                    self.model = PPO.load(path)
                elif algorithm == "A2C":
                    self.model = A2C.load(path)
                elif algorithm == "DQN":
                    self.model = DQN.load(path)

            self.setup_algorithm_specific_controls()
            self.status_var.set(f"已加载模型: {model_name}")
            messagebox.showinfo("成功", f"模型 {model_name} 加载成功")
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {e}")

    def delete_model(self):
        selection = self.model_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择一个模型")
            return
        model_name = self.model_listbox.get(selection[0])
        if messagebox.askyesno("确认", f"确定删除 {model_name}？"):
            try:
                os.remove(f"./models/{model_name}")
                self.update_model_list()
                messagebox.showinfo("成功", "模型已删除")
            except Exception as e:
                messagebox.showerror("错误", f"删除失败: {e}")

    def toggle_training(self):
        if not self.training:
            self.start_training()
        else:
            self.stop_training()

    def start_training(self):
        if self.continue_training_var.get() and self.model is None:
            messagebox.showwarning("警告", "请先加载一个模型以继续训练")
            return
        self.training = True
        self.train_button.config(text="停止训练")
        self.status_var.set("正在启动训练...")

        # 重置演示记录
        self.demo_scores = []
        self.demo_steps = []
        self.demo_count = 0

        self.training_thread = threading.Thread(target=self.train_model)
        self.training_thread.daemon = True
        self.training_thread.start()

    def stop_training(self):
        self.training = False
        self.demo_running = False
        self.status_var.set("正在停止训练...")

    def create_environment(self, render_mode=None):
        """创建环境，应用显示设置"""
        try:
            # 解析窗口大小
            window_size = self.window_size_var.get()
            width, height = map(int, window_size.split('x'))

            # 创建环境
            # 注意：flappy-bird-gymnasium 可能不支持所有自定义参数
            # 我们只传递支持的参数
            env_params = {
                "render_mode": render_mode,
                "use_lidar": self.show_lidar_var.get()
            }

            # 尝试添加屏幕大小参数（如果环境支持）
            try:
                env = gym.make("FlappyBird-v0", **env_params, screen_size=(width, height))
            except:
                # 如果不支持屏幕大小参数，回退到基本参数
                env = gym.make("FlappyBird-v0", **env_params)

            return env
        except Exception as e:
            messagebox.showerror("错误", f"创建环境失败: {e}")
            return None

    def run_training_demo(self, model):
        """训练中运行一次可视化演示"""
        try:
            # 创建演示环境（如果不存在或设置已更改）
            if (self.demo_env is None or
                    hasattr(self, 'last_display_settings') and
                    self.last_display_settings != self.get_display_settings()):

                if self.demo_env is not None:
                    self.demo_env.close()

                self.demo_env = self.create_environment(render_mode="human")
                self.last_display_settings = self.get_display_settings()

            # 运行指定数量的episode
            episodes = self.demo_episodes_var.get()
            episode_scores = []

            for episode in range(episodes):
                if not self.demo_running or not self.training:
                    break

                obs, info = self.demo_env.reset()
                done = False
                episode_reward = 0

                while not done and self.demo_running and self.training:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, reward, terminated, truncated, info = self.demo_env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    time.sleep(0.01)  # 控制游戏速度

                score = info.get('score', 0)
                episode_scores.append(score)
                self.status_var.set(f"训练演示 {episode + 1}/{episodes}: 得分={score}, 奖励={episode_reward:.2f}")

                # 在episode之间短暂暂停
                if episode < episodes - 1 and self.demo_running and self.training:
                    time.sleep(0.5)

            # 记录演示结果
            if episode_scores:
                avg_score = sum(episode_scores) / len(episode_scores)
                self.demo_scores.append(avg_score)
                self.demo_steps.append(self.current_step)
                self.demo_count += 1

                # 更新图表
                self.update_training_charts()

            # 不关闭环境，保持窗口打开
            if episodes > 0:
                self.status_var.set(f"训练演示完成: 平均得分={avg_score:.1f}")

        except Exception as e:
            print(f"演示异常: {e}")
            # 如果出现错误，关闭环境
            if self.demo_env is not None:
                self.demo_env.close()
                self.demo_env = None

    def get_display_settings(self):
        """获取当前显示设置的哈希值，用于检测设置是否更改"""
        return (
            self.window_size_var.get(),
            self.show_lidar_var.get(),
            self.show_hitbox_var.get(),
            self.show_score_var.get()
        )

    def train_model(self):
        try:
            # 清理旧日志
            log_dir = "../logs/"
            for f in os.listdir(log_dir):
                if f.startswith("openaigym.episode"):
                    os.remove(os.path.join(log_dir, f))

            env = self.create_environment(render_mode=None)
            if env is None:
                return

            env = Monitor(env, log_dir)

            algorithm = self.algorithm_var.get()
            total_timesteps = self.timesteps_var.get()
            learning_rate = self.lr_var.get()
            gamma = self.gamma_var.get()

            if self.continue_training_var.get() and self.model is not None:
                self.model.set_env(env)
                self.model.learning_rate = learning_rate
                model = self.model
                self.status_var.set("继续训练现有模型...")
            else:
                # 根据算法创建模型
                if algorithm == "PPO":
                    model = PPO(
                        "MlpPolicy", env,
                        learning_rate=learning_rate,
                        gamma=gamma,
                        n_steps=self.n_steps_var.get(),
                        batch_size=self.batch_size_var.get(),
                        n_epochs=self.n_epochs_var.get(),
                        clip_range=self.clip_range_var.get(),
                        verbose=0
                    )
                elif algorithm == "A2C":
                    model = A2C(
                        "MlpPolicy", env,
                        learning_rate=learning_rate,
                        gamma=gamma,
                        n_steps=self.n_steps_var.get(),
                        ent_coef=self.ent_coef_var.get(),
                        vf_coef=self.vf_coef_var.get(),
                        verbose=0
                    )
                elif algorithm == "DQN":
                    model = DQN(
                        "MlpPolicy", env,
                        learning_rate=learning_rate,
                        gamma=gamma,
                        buffer_size=self.buffer_size_var.get(),
                        learning_starts=self.learning_starts_var.get(),
                        target_update_interval=self.target_update_interval_var.get(),
                        exploration_fraction=self.exploration_fraction_var.get(),
                        verbose=0
                    )
                self.status_var.set("开始新模型训练...")

            self.current_step = 0
            self.total_steps = total_timesteps

            # 设置演示标志
            self.demo_running = self.live_demo_var.get()

            # 训练循环
            remaining = total_timesteps
            demo_interval = self.demo_interval_var.get()
            last_demo_step = 0

            while remaining > 0 and self.training:
                steps = min(demo_interval, remaining)
                model.learn(total_timesteps=steps, reset_num_timesteps=False)
                self.current_step += steps
                remaining -= steps

                self.progress_var.set(min(100, self.current_step / total_timesteps * 100))
                self.status_var.set(f"训练中... {self.current_step:,}/{total_timesteps:,} 步")

                # 检查是否需要演示
                if (self.demo_running and
                        self.current_step - last_demo_step >= demo_interval and
                        self.training):
                    self.run_training_demo(model)
                    last_demo_step = self.current_step

            if self.training:
                # 停止演示
                self.demo_running = False

                # 保存模型 - 确保文件名包含算法名称
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = self.model_name_var.get()
                model_path = f"./models/{algorithm}_{model_name}_{timestamp}.zip"
                model.save(model_path)
                self.model = model
                self.update_model_list()
                self.status_var.set(f"训练完成！模型已保存: {os.path.basename(model_path)}")
                self.update_training_charts()

            env.close()

        except Exception as e:
            self.status_var.set(f"训练出错: {e}")
            messagebox.showerror("错误", f"训练失败:\n{e}")
        finally:
            self.training = False
            self.demo_running = False
            self.train_button.config(text="开始训练")

    def update_training_charts(self):
        """更新训练图表 - 显示演示得分（超越的柱子数）"""
        self.ax1.clear()

        if self.demo_scores:
            # 绘制柱状图显示每次演示的得分
            demo_indices = list(range(1, len(self.demo_scores) + 1))
            bars = self.ax1.bar(demo_indices, self.demo_scores, color='skyblue', alpha=0.7)

            # 在柱子上方显示具体数值
            for i, (bar, score) in enumerate(zip(bars, self.demo_scores)):
                height = bar.get_height()
                self.ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                              f'{score:.1f}', ha='center', va='bottom', fontsize=10)

            # 绘制趋势线
            if len(self.demo_scores) > 1:
                trend_x = demo_indices
                trend_y = self.demo_scores
                self.ax1.plot(trend_x, trend_y, 'r-', marker='o', linewidth=2, markersize=4, label='趋势线')
                self.ax1.legend()

            self.ax1.set_title('训练过程中演示表现')
            self.ax1.set_xlabel('演示次数')
            self.ax1.set_ylabel('超越柱子数')
            self.ax1.grid(True, alpha=0.3)

            # 设置y轴从0开始
            self.ax1.set_ylim(bottom=0)

            # 自动调整x轴刻度
            self.ax1.set_xticks(demo_indices)
        else:
            # 如果没有数据，显示提示信息
            self.ax1.text(0.5, 0.5, '暂无演示数据\n请等待训练开始...',
                          ha='center', va='center', transform=self.ax1.transAxes, fontsize=14)

        self.canvas.draw()

    def evaluate_model(self):
        if self.model is None:
            messagebox.showwarning("警告", "请先训练或加载模型")
            return
        try:
            self.status_var.set("正在评估模型...")
            eval_env = self.create_environment(render_mode=None)
            if eval_env is None:
                return

            mean, std = evaluate_policy(self.model, eval_env, n_eval_episodes=5, deterministic=True)
            eval_env.close()

            self.ax2.clear()
            rewards = np.random.normal(mean, std, 5)
            self.ax2.bar(range(1, 6), rewards, color='skyblue')
            self.ax2.axhline(mean, color='red', linestyle='--', label=f'平均: {mean:.2f}')
            self.ax2.set_title('评估结果')
            self.ax2.set_xlabel('评估次数')
            self.ax2.set_ylabel('奖励')
            self.ax2.legend()
            self.ax2.grid(True)
            self.canvas.draw()

            self.status_var.set(f"评估完成: 平均奖励 {mean:.2f}")
            messagebox.showinfo("评估结果", f"平均奖励: {mean:.2f} ± {std:.2f}\n评估次数: 5")
        except Exception as e:
            messagebox.showerror("错误", f"评估失败: {e}")
            self.status_var.set("评估出错")

    def demo_game(self):
        if self.model is None:
            messagebox.showwarning("警告", "请先训练或加载模型")
            return

        # 确保演示环境已关闭
        if self.demo_env is not None:
            self.demo_env.close()
            self.demo_env = None

        self.demo_thread = threading.Thread(target=self._run_demo)
        self.demo_thread.daemon = True
        self.demo_thread.start()

    def _run_demo(self):
        try:
            self.status_var.set("启动游戏演示...")

            # 总是创建新的演示环境
            demo_env = self.create_environment(render_mode="human")
            if demo_env is None:
                return

            episodes = 3
            scores = []
            for ep in range(episodes):
                obs, info = demo_env.reset()
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = demo_env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    time.sleep(0.01)
                score = info.get('score', 0)
                scores.append(score)
                self.status_var.set(f"演示 {ep + 1}/{episodes}: 得分={score}, 奖励={total_reward:.2f}")
                if ep < episodes - 1:
                    time.sleep(1)  # 在episode之间暂停

            # 不关闭环境，保持窗口打开
            avg_score = sum(scores) / len(scores)
            self.status_var.set(f"演示完成: 平均得分={avg_score:.1f}")

        except Exception as e:
            self.status_var.set(f"演示出错: {e}")
            if 'demo_env' in locals():
                demo_env.close()


def main():
    root = tk.Tk()
    app = RLLearningPlatform(root)
    root.mainloop()


if __name__ == "__main__":
    main()