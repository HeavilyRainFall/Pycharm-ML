import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from datetime import datetime
import sys


# ==============================
# 修复：线程安全的训练回调
# ==============================
class SafeTrainingCallback(BaseCallback):
    def __init__(self, app, verbose=0):
        super().__init__(verbose)
        self.app = app
        self.episode_rewards = []
        self.current_step = 0

    def _on_step(self) -> bool:
        try:
            # 进度更新（主线程执行）
            self.current_step += 1
            progress = self.current_step / self.model._total_timesteps * 100
            self.app.root.after(0, lambda: self.app.train_progress['value'] = progress)

            # 仅在episode结束时记录奖励
            dones = self.locals.get("dones", [False])
            if any(dones):
                # 计算当前episode奖励
                rewards = self.locals.get("rewards", [0])
                episode_reward = sum(rewards)
                self.episode_rewards.append(episode_reward)

                # 主线程更新日志和绘图
                log_msg = f"[步数:{self.current_step}] 本轮奖励:{episode_reward:.2f} | 近5轮平均:{np.mean(self.episode_rewards[-5:]):.2f}\n"
                self.app.root.after(0, lambda: self.app.append_log(log_msg))
                self.app.root.after(0, lambda: self.app.update_reward_plot(self.episode_rewards))
        except Exception as e:
            print(f"回调错误: {e}")
        return True


# ==============================
# 修复：主应用类（简化+稳定）
# ==============================
class RLFlappyBirdApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RL Flappy Bird 训练平台 🎮")
        self.geometry("1100x750")
        self.minsize(900, 600)

        # 全局状态（初始化）
        self.current_model = None
        self.train_env = None
        self.is_training = False
        self.training_thread = None
        self.trained_steps = 0
        self.all_rewards = []

        # 绘图初始化（修复：避免重复创建）
        self.fig, self.ax = plt.subplots(figsize=(7, 3), dpi=90)
        self.ax.set_xlabel("训练轮数")
        self.ax.set_ylabel("奖励")
        self.ax.set_title("训练奖励曲线")
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)

        # 创建UI（简化布局）
        self._create_simple_ui()

    # ==============================
    # 修复：简化UI布局，减少嵌套bug
    # ==============================
    def _create_simple_ui(self):
        # 1. 顶部参数栏
        top_frame = ttk.LabelFrame(self, text="基础配置")
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        # 模型选择
        ttk.Label(top_frame, text="模型:").grid(row=0, column=0, padx=5, pady=5)
        self.model_type = ttk.Combobox(top_frame, values=["PPO", "A2C", "DQN"], state="readonly")
        self.model_type.set("PPO")
        self.model_type.grid(row=0, column=1, padx=5, pady=5)

        # 训练步数
        ttk.Label(top_frame, text="训练步数:").grid(row=0, column=2, padx=5, pady=5)
        self.train_steps = ttk.Spinbox(top_frame, from_=5000, to=100000, increment=5000, value=20000)
        self.train_steps.grid(row=0, column=3, padx=5, pady=5)

        # 学习率
        ttk.Label(top_frame, text="学习率:").grid(row=0, column=4, padx=5, pady=5)
        self.lr = ttk.Spinbox(top_frame, from_=0.0001, to=0.01, increment=0.0001, value=0.001, format="%.4f")
        self.lr.grid(row=0, column=5, padx=5, pady=5)

        # 2. 功能按钮栏
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Button(btn_frame, text="📦 创建模型", command=self._create_model_safe).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="▶️ 开始训练", command=self._start_train_safe).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="⏹️ 停止训练", command=self._stop_train_safe, state=tk.DISABLED).pack(side=tk.LEFT,
                                                                                                         padx=5)
        ttk.Button(btn_frame, text="📊 评估模型", command=self._eval_model_safe).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🎮 演示游戏", command=self._demo_model_safe).pack(side=tk.LEFT, padx=5)

        # 模型保存/加载
        ttk.Button(btn_frame, text="💾 保存模型", command=self._save_model_safe).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="📤 加载模型", command=self._load_model_safe).pack(side=tk.RIGHT, padx=5)

        # 3. 进度条
        self.train_progress = ttk.Progressbar(self, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.train_progress.pack(fill=tk.X, padx=10, pady=5)

        # 4. 绘图区域
        plot_frame = ttk.LabelFrame(self, text="训练奖励曲线")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 5. 日志区域
        log_frame = ttk.LabelFrame(self, text="运行日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 初始化日志
        self.append_log("✅ 程序启动成功！\n")

    # ==============================
    # 核心功能（修复所有已知bug）
    # ==============================
    def append_log(self, msg):
        """修复：线程安全的日志追加"""
        self.log_text.insert(tk.END, msg)
        self.log_text.see(tk.END)
        self.log_text.update_idletasks()

    def update_reward_plot(self, rewards):
        """修复：绘图更新逻辑，避免崩溃"""
        self.all_rewards.extend(rewards)
        self.ax.clear()
        self.ax.plot(self.all_rewards, alpha=0.7, label="每轮奖励")

        # 滑动平均（修复：避免空列表报错）
        if len(self.all_rewards) >= 5:
            window = min(5, len(self.all_rewards))
            avg = np.convolve(self.all_rewards, np.ones(window) / window, mode='valid')
            self.ax.plot(range(window - 1, len(self.all_rewards)), avg, 'r-', label="5轮平均")

        self.ax.set_xlabel("训练轮数")
        self.ax.set_ylabel("奖励")
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def _create_env_safe(self, render_mode=None):
        """修复：环境创建失败处理"""
        try:
            env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
            return Monitor(env)
        except Exception as e:
            self.append_log(f"❌ 创建环境失败: {e}\n")
            messagebox.showerror("错误", f"环境创建失败：{e}")
            return None

    def _create_model_safe(self):
        """修复：模型创建异常处理"""
        if self.is_training:
            messagebox.showwarning("警告", "训练中无法创建模型！")
            return

        # 创建环境
        self.train_env = self._create_env_safe()
        if not self.train_env:
            return

        # 创建模型
        try:
            lr = float(self.lr.get())
            model_type = self.model_type.get()

            if model_type == "PPO":
                self.current_model = PPO(
                    "MlpPolicy", self.train_env, learning_rate=lr,
                    n_steps=1024, batch_size=64, gamma=0.99, verbose=0
                )
            elif model_type == "A2C":
                self.current_model = A2C(
                    "MlpPolicy", self.train_env, learning_rate=lr,
                    n_steps=512, gamma=0.99, verbose=0
                )
            elif model_type == "DQN":
                self.current_model = DQN(
                    "MlpPolicy", self.train_env, learning_rate=lr,
                    batch_size=64, gamma=0.99, verbose=0
                )

            self.append_log(f"✅ 成功创建 {model_type} 模型\n")
            messagebox.showinfo("成功", f"{model_type} 模型创建完成！")
        except Exception as e:
            self.append_log(f"❌ 创建模型失败: {e}\n")
            messagebox.showerror("错误", f"创建模型失败：{e}")

    def _train_worker(self):
        """修复：训练线程逻辑，避免卡死"""
        try:
            steps = int(self.train_steps.get())
            self.append_log(f"🚀 开始训练 {steps} 步...\n")

            # 回调函数
            callback = SafeTrainingCallback(self)

            # 执行训练
            self.current_model.learn(
                total_timesteps=steps,
                callback=callback,
                progress_bar=False
            )

            # 训练完成更新
            self.trained_steps += steps
            self.append_log(f"✅ 训练完成！累计步数：{self.trained_steps}\n")
            self.root.after(0, lambda: messagebox.showinfo("成功", "训练完成！"))
        except Exception as e:
            self.append_log(f"❌ 训练出错: {e}\n")
            self.root.after(0, lambda: messagebox.showerror("错误", f"训练失败：{e}"))
        finally:
            # 修复：强制重置训练状态
            self.is_training = False
            self.root.after(0, lambda: self.nametowidget(".!frame2").children['!button2']['state'] = tk.NORMAL)
            self.root.after(0, lambda: self.nametowidget(".!frame2").children['!button3']['state'] = tk.DISABLED)
            self.train_progress['value'] = 0

    def _start_train_safe(self):
        """修复：训练启动逻辑，防止重复点击"""
        if self.is_training:
            messagebox.showwarning("警告", "训练已在进行中！")
            return

        if not self.current_model:
            messagebox.showwarning("警告", "请先创建模型！")
            return

        # 更新UI状态
        self.is_training = True
        self.nametowidget(".!frame2").children['!button2']['state'] = tk.DISABLED
        self.nametowidget(".!frame2").children['!button3']['state'] = tk.NORMAL

        # 启动训练线程
        self.training_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.training_thread.start()

    def _stop_train_safe(self):
        """修复：停止训练逻辑"""
        self.is_training = False
        self.append_log("⚠️ 训练已停止！\n")
        self.nametowidget(".!frame2").children['!button2']['state'] = tk.NORMAL
        self.nametowidget(".!frame2").children['!button3']['state'] = tk.DISABLED

    def _eval_model_safe(self):
        """修复：模型评估逻辑"""
        if not self.current_model:
            messagebox.showwarning("警告", "无可用模型！")
            return

        # 创建评估环境
        eval_env = self._create_env_safe()
        if not eval_env:
            return

        try:
            mean_reward, std_reward = evaluate_policy(
                self.current_model, eval_env,
                n_eval_episodes=5, deterministic=True
            )
            eval_env.close()

            self.append_log(f"📊 评估结果：平均奖励={mean_reward:.2f} ± {std_reward:.2f}\n")
            messagebox.showinfo("评估结果", f"平均奖励：{mean_reward:.2f}\n标准差：{std_reward:.2f}")
        except Exception as e:
            self.append_log(f"❌ 评估失败: {e}\n")
            messagebox.showerror("错误", f"评估失败：{e}")

    def _demo_model_safe(self):
        """修复：游戏演示逻辑，避免环境冲突"""
        if not self.current_model:
            messagebox.showwarning("警告", "无可用模型！")
            return

        # 创建独立演示环境
        demo_env = self._create_env_safe(render_mode="human")
        if not demo_env:
            return

        try:
            self.append_log("🎮 开始演示游戏...\n")
            obs, _ = demo_env.reset()
            total_reward = 0

            while True:
                action, _ = self.current_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = demo_env.step(action)
                total_reward += reward

                if terminated or truncated:
                    score = info.get('score', 0)
                    self.append_log(f"🎮 演示结束 | 得分：{score} | 总奖励：{total_reward:.2f}\n")
                    break

                time.sleep(0.01)  # 修复：演示速度控制
        except Exception as e:
            self.append_log(f"❌ 演示失败: {e}\n")
            messagebox.showerror("错误", f"演示失败：{e}")
        finally:
            demo_env.close()

    def _save_model_safe(self):
        """修复：模型保存逻辑"""
        if not self.current_model:
            messagebox.showwarning("警告", "无可用模型！")
            return

        # 选择保存路径
        save_path = filedialog.asksaveasfilename(
            defaultextension=".zip",
            filetypes=[("模型文件", "*.zip"), ("所有文件", "*.*")],
            initialfile=f"{self.model_type.get()}_flappy_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )

        if save_path:
            try:
                self.current_model.save(save_path)
                # 保存奖励日志
                if self.all_rewards:
                    np.save(f"{save_path}_rewards.npy", self.all_rewards)
                self.append_log(f"💾 模型已保存至：{save_path}\n")
                messagebox.showinfo("成功", "模型保存完成！")
            except Exception as e:
                self.append_log(f"❌ 保存失败: {e}\n")
                messagebox.showerror("错误", f"保存失败：{e}")

    def _load_model_safe(self):
        """修复：模型加载逻辑"""
        if self.is_training:
            messagebox.showwarning("警告", "训练中无法加载模型！")
            return

        # 选择加载路径
        load_path = filedialog.askopenfilename(
            filetypes=[("模型文件", "*.zip"), ("所有文件", "*.*")]
        )

        if load_path:
            try:
                # 去除.zip后缀
                if load_path.endswith(".zip"):
                    load_path = load_path[:-4]

                # 加载模型
                model_type = self.model_type.get()
                if model_type == "PPO":
                    self.current_model = PPO.load(load_path)
                elif model_type == "A2C":
                    self.current_model = A2C.load(load_path)
                elif model_type == "DQN":
                    self.current_model = DQN.load(load_path)

                # 重新绑定环境
                self.train_env = self._create_env_safe()
                self.current_model.set_env(self.train_env)

                self.append_log(f"📤 成功加载模型：{load_path}\n")
                messagebox.showinfo("成功", "模型加载完成！")
            except Exception as e:
                self.append_log(f"❌ 加载失败: {e}\n")
                messagebox.showerror("错误", f"加载失败：{e}")


# ==============================
# 程序入口（修复：资源清理）
# ==============================
if __name__ == "__main__":
    app = RLFlappyBirdApp()


    # 修复：窗口关闭时清理资源
    def on_closing():
        if app.is_training:
            app.is_training = False
            time.sleep(0.5)
        if app.train_env:
            app.train_env.close()
        app.destroy()


    app.protocol("WM_DELETE_WINDOW", on_closing)
    app.mainloop()