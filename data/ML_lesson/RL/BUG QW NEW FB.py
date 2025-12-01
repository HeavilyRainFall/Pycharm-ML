import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# 全局变量：用于跨回调传递奖励
training_rewards = []
training_steps = []


class RewardLogger:
    def __init__(self):
        self.rewards = []
        self.steps = []

    def log(self, reward, step):
        self.rewards.append(reward)
        self.steps.append(step)


reward_logger = RewardLogger()


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # 执行评估
            results = self._run_eval()
            if results:
                avg_reward = results['eval/mean_reward']
                total_steps = self.model.num_timesteps
                self.logger.log(avg_reward, total_steps)
        return True


class RLTrainingPlatform:
    def __init__(self, root):
        self.root = root
        self.root.title("Flappy Bird 强化学习训练平台")
        self.root.geometry("900x700")

        self.model = None
        self.training = False
        self.train_rewards = []
        self.train_steps_recorded = []

        self.setup_ui()
        self.load_model_if_exists()

    def setup_ui(self):
        # 控制区
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)

        self.train_button = ttk.Button(control_frame, text="开始训练", command=self.start_training)
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.demo_button = ttk.Button(control_frame, text="演示游戏", command=self.demo_game)
        self.demo_button.pack(side=tk.LEFT, padx=5)

        self.load_button = ttk.Button(control_frame, text="加载模型", command=self.load_model)
        self.load_button.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="就绪")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=10)

        # 图表区
        chart_frame = ttk.Frame(self.root)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title("训练奖励曲线")
        self.ax.set_xlabel("训练步数")
        self.ax.set_ylabel("平均奖励")
        self.ax.grid(True)
        self.line, = self.ax.plot([], [], 'b-o', markersize=3)
        self.canvas = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_environment(self, render_mode=None):
        try:
            env = gym.make("FlappyBird-v0", render_mode=render_mode)
            return env
        except Exception as e:
            messagebox.showerror("环境错误", f"无法创建环境: {e}")
            return None

    def load_model_if_exists(self):
        if os.path.exists("flappy_ppo.zip"):
            try:
                self.model = PPO.load("flappy_ppo.zip")
                self.status_var.set("模型已加载")
            except Exception as e:
                self.status_var.set(f"加载模型失败: {e}")

    def start_training(self):
        if self.training:
            messagebox.showinfo("提示", "训练已在进行中")
            return
        self.training = True
        self.train_button.config(text="训练中...")
        threading.Thread(target=self.train_model, daemon=True).start()

    def train_model(self):
        global training_rewards, training_steps
        training_rewards.clear()
        training_steps.clear()
        reward_logger.rewards.clear()
        reward_logger.steps.clear()

        try:
            env = self.create_environment(render_mode=None)
            if env is None:
                return

            eval_env = self.create_environment(render_mode=None)
            if eval_env is None:
                env.close()
                return

            # 使用自定义回调记录评估奖励
            eval_callback = CustomEvalCallback(
                eval_env,
                best_model_save_path="./logs/",
                log_path="./logs/",
                eval_freq=1000,  # 每1000步评估一次
                deterministic=True,
                render=False,
                n_eval_episodes=3,
                logger=reward_logger
            )

            self.model = PPO("MlpPolicy", env, verbose=0, n_steps=512, learning_rate=0.0003)

            self.status_var.set("开始训练...")
            self.model.learn(total_timesteps=50000, callback=eval_callback)

            # 保存模型
            self.model.save("flappy_ppo.zip")
            self.status_var.set("训练完成，模型已保存")

            # 更新图表数据
            self.train_rewards = reward_logger.rewards[:]
            self.train_steps_recorded = reward_logger.steps[:]
            self.update_training_charts()

        except Exception as e:
            self.status_var.set(f"训练出错: {e}")
            messagebox.showerror("训练错误", str(e))
        finally:
            if 'env' in locals():
                env.close()
            if 'eval_env' in locals():
                eval_env.close()
            self.training = False
            self.train_button.config(text="开始训练")

    def update_training_charts(self):
        if not self.train_rewards:
            self.ax.clear()
            self.ax.text(0.5, 0.5, '暂无训练数据', horizontalalignment='center', verticalalignment='center',
                         transform=self.ax.transAxes)
        else:
            self.ax.clear()
            self.ax.plot(self.train_steps_recorded, self.train_rewards, 'b-o', markersize=3)
            self.ax.set_title("训练奖励曲线")
            self.ax.set_xlabel("训练步数")
            self.ax.set_ylabel("平均奖励")
            self.ax.grid(True)
        self.canvas.draw()

    def demo_game(self):
        if self.model is None:
            messagebox.showwarning("警告", "请先训练或加载模型")
            return
        threading.Thread(target=self._run_demo, daemon=True).start()

    def _run_demo(self):
        try:
            self.status_var.set("启动游戏演示...")
            # 每次都新建环境，不在子线程复用旧环境
            demo_env = self.create_environment(render_mode="human")
            if demo_env is None:
                return
            episodes = 3
            for ep in range(episodes):
                obs, info = demo_env.reset()
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = demo_env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    time.sleep(0.01)  # 控制速度
                score = info.get('score', 0)
                self.status_var.set(f"演示 {ep + 1}/{episodes}: 得分={score}, 奖励={total_reward:.2f}")
                if ep < episodes - 1:
                    time.sleep(1)
            demo_env.close()
            self.status_var.set("演示完成")
        except Exception as e:
            self.status_var.set(f"演示出错: {e}")
            messagebox.showerror("演示错误", str(e))

    def load_model(self):
        if os.path.exists("flappy_ppo.zip"):
            try:
                self.model = PPO.load("flappy_ppo.zip")
                self.status_var.set("模型加载成功")
            except Exception as e:
                self.status_var.set(f"加载失败: {e}")
        else:
            messagebox.showwarning("文件不存在", "未找到 flappy_ppo.zip 模型文件")


if __name__ == "__main__":
    root = tk.Tk()
    app = RLTrainingPlatform(root)
    root.mainloop()