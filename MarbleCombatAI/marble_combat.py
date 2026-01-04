import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import math
from pygame import gfxdraw


class SingleAgentWrapper(gym.Env):
    """
    将双智能体环境包装为单智能体环境，以便与Stable-Baselines3兼容
    """
    def __init__(self, base_env, agent_id='attacker'):
        super().__init__()
        self.base_env = base_env
        self.agent_id = agent_id  # 'attacker' 或 'defender'
        
        # 根据智能体ID设置动作和观测空间
        if agent_id == 'attacker':
            self.action_space = base_env.action_space['attacker']
            self.observation_space = base_env.observation_space['attacker']
        else:  # defender
            self.action_space = base_env.action_space['defender']
            self.observation_space = base_env.observation_space['defender']
        
        # 保存其他环境属性
        self.render_mode = base_env.render_mode
        self.width = base_env.width
        self.height = base_env.height
        
    def reset(self, seed=None):
        obs, info = self.base_env.reset(seed=seed)
        return obs[self.agent_id], info

    def step(self, action):
        # 需要获取另一个智能体的动作，这里使用随机动作作为示例
        if self.agent_id == 'attacker':
            # 获取防守方的随机动作
            defender_action = self.base_env.action_space['defender'].sample()
            actions = {
                'attacker': action,
                'defender': defender_action
            }
        else:  # defender
            # 获取攻击方的随机动作
            attacker_action = self.base_env.action_space['attacker'].sample()
            actions = {
                'attacker': attacker_action,
                'defender': action
            }
        
        obs, rewards, terminated, truncated, info = self.base_env.step(actions)
        return obs[self.agent_id], rewards[self.agent_id], terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def close(self):
        self.base_env.close()


class MarbleCombatEnv(gym.Env):
    """
    弹珠对抗环境
    两个智能体控制弹珠进行对抗：
    - 智能体A：攻击方，目标是发射弹珠打中目标（砖块）
    - 智能体B：防守方，目标是守门，防止对方得分
    """
    
    def __init__(self, render_mode=None, size=800):
        super().__init__()
        
        self.size = size  # 环境大小 800x600
        self.width = 800
        self.height = 600
        self.window_size = 800  # 渲染窗口大小
        
        # 动作空间：[力度控制, 角度控制] 或 [x方向力, y方向力]
        # 攻击方：控制发射力度和角度
        # 防守方：控制守门员位置
        self.action_space = spaces.Dict({
            'attacker': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),  # 发射力度和角度
            'defender': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)    # 守门员位置 (x, y)
        })
        
        # 观察空间：每个智能体可以看到球的位置、速度和目标位置
        self.observation_space = spaces.Dict({
            'attacker': spaces.Box(low=0, high=float(self.width), shape=(6,), dtype=np.float32),  # 球位置、速度、目标位置
            'defender': spaces.Box(low=0, high=float(self.height), shape=(6,), dtype=np.float32)   # 球位置、速度、门位置
        })
        
        self.render_mode = render_mode
        
        # 初始化pygame
        if render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont(None, 24)
            
        # 游戏元素
        self.ball_pos = [self.width // 2, self.height // 2]
        self.ball_vel = [0.0, 0.0]
        self.ball_radius = 10
        self.ball_mass = 1.0
        
        # 攻击方弹珠
        self.attacker_marble_pos = [50, self.height // 2]
        self.attacker_marble_vel = [0.0, 0.0]
        self.attacker_marble_radius = 15
        
        # 防守方守门员
        self.defender_pos = [self.width - 50, self.height // 2]
        self.defender_radius = 20
        
        # 目标区域（门）
        self.goal_top = self.height // 2 - 100
        self.goal_bottom = self.height // 2 + 100
        self.goal_width = 20
        
        # 砖块位置（可扩展）
        self.bricks = [
            [self.width - 150, 100],
            [self.width - 150, 150],
            [self.width - 150, 200],
            [self.width - 150, self.height - 100],
            [self.width - 150, self.height - 150],
            [self.width - 150, self.height - 200]
        ]
        
        # 游戏状态
        self.score_a = 0  # 攻击方得分
        self.score_b = 0  # 防守方得分
        self.max_steps = 1000
        self.current_step = 0
        
        # 物理参数
        self.gravity = 0.0  # 移除重力，让弹珠更像台球
        self.friction = 0.99
        self.restitution = 0.8  # 弹性系数

    def _get_obs(self):
        """获取当前观测"""
        # 攻击方观测：球位置、速度、目标位置
        obs_a = np.array([
            self.attacker_marble_pos[0] / self.width,
            self.attacker_marble_pos[1] / self.height,
            self.attacker_marble_vel[0],
            self.attacker_marble_vel[1],
            (self.width - 150) / self.width,  # 砖块x坐标
            (self.height // 2) / self.height  # 中央y坐标作为参考
        ], dtype=np.float32)
        
        # 防守方观测：球位置、速度、门位置
        obs_b = np.array([
            self.attacker_marble_pos[0] / self.width,
            self.attacker_marble_pos[1] / self.height,
            self.attacker_marble_vel[0],
            self.attacker_marble_vel[1],
            (self.width - self.goal_width) / self.width,  # 门的x坐标
            (self.height // 2) / self.height  # 门中央y坐标
        ], dtype=np.float32)
        
        return {'attacker': obs_a, 'defender': obs_b}

    def _get_info(self):
        """获取额外信息"""
        return {
            "ball_pos": self.attacker_marble_pos,
            "ball_vel": self.attacker_marble_vel,
            "defender_pos": self.defender_pos,
            "score_a": self.score_a,
            "score_b": self.score_b,
            "step": self.current_step
        }

    def reset(self, seed=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 重置球的位置和速度
        self.attacker_marble_pos = [50, self.height // 2]
        self.attacker_marble_vel = [0.0, 0.0]
        self.defender_pos = [self.width - 50, self.height // 2]
        
        # 重置分数
        self.score_a = 0
        self.score_b = 0
        self.current_step = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, info

    def step(self, actions):
        """执行一步操作"""
        self.current_step += 1
        
        # 处理攻击方动作：控制发射力度和角度
        att_action = actions['attacker']
        power = np.clip(att_action[0], -1.0, 1.0) * 10.0  # 力度
        angle = np.clip(att_action[1], -1.0, 1.0) * np.pi  # 角度
        
        # 如果球在初始位置，应用发射力
        if (abs(self.attacker_marble_pos[0] - 50) < 5 and 
            abs(self.attacker_marble_pos[1] - self.height // 2) < 5):
            self.attacker_marble_vel[0] = power * np.cos(angle)
            self.attacker_marble_vel[1] = power * np.sin(angle)
        
        # 处理防守方动作：控制守门员位置
        def_action = actions['defender']
        # 守门员在球门区域内移动
        self.defender_pos[0] = self.width - 50  # 固定x位置
        self.defender_pos[1] = np.clip(
            def_action[1] * self.height, 
            self.goal_top + self.defender_radius, 
            self.goal_bottom - self.defender_radius
        )  # y位置在球门范围内
        
        # 更新物理
        self._update_physics()
        
        # 检查碰撞和得分
        rewards = self._check_collisions()
        
        # 检查是否结束
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        observations = self._get_obs()
        infos = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observations, rewards, terminated, truncated, infos

    def _update_physics(self):
        """更新物理模拟"""
        # 更新攻击方弹珠位置
        self.attacker_marble_pos[0] += self.attacker_marble_vel[0]
        self.attacker_marble_pos[1] += self.attacker_marble_vel[1]
        
        # 应用重力（轻微重力，使弹珠更可控）
        self.attacker_marble_vel[1] += self.gravity
        
        # 应用摩擦力
        self.attacker_marble_vel[0] *= self.friction
        self.attacker_marble_vel[1] *= self.friction
        
        # 边界碰撞检测
        if self.attacker_marble_pos[0] <= self.attacker_marble_radius:
            self.attacker_marble_pos[0] = self.attacker_marble_radius
            self.attacker_marble_vel[0] *= -self.restitution
        elif self.attacker_marble_pos[0] >= self.width - self.attacker_marble_radius:
            self.attacker_marble_pos[0] = self.width - self.attacker_marble_radius
            self.attacker_marble_vel[0] *= -self.restitution
            
        if self.attacker_marble_pos[1] <= self.attacker_marble_radius:
            self.attacker_marble_pos[1] = self.attacker_marble_radius
            self.attacker_marble_vel[1] *= -self.restitution
        elif self.attacker_marble_pos[1] >= self.height - self.attacker_marble_radius:
            self.attacker_marble_pos[1] = self.height - self.attacker_marble_radius
            self.attacker_marble_vel[1] *= -self.restitution

    def _check_collisions(self):
        """检查碰撞并计算奖励"""
        rewards = {'attacker': 0.0, 'defender': 0.0}
        
        # 检查与守门员的碰撞
        dist_to_defender = math.sqrt(
            (self.attacker_marble_pos[0] - self.defender_pos[0])**2 +
            (self.attacker_marble_pos[1] - self.defender_pos[1])**2
        )
        
        if dist_to_defender <= (self.attacker_marble_radius + self.defender_radius):
            # 球与守门员碰撞
            rewards['defender'] += 5.0  # 防守方得分
            rewards['attacker'] -= 2.0  # 攻击方失分
            
            # 简单的碰撞反弹
            dx = self.attacker_marble_pos[0] - self.defender_pos[0]
            dy = self.attacker_marble_pos[1] - self.defender_pos[1]
            distance = max(0.1, math.sqrt(dx*dx + dy*dy))
            nx, ny = dx/distance, dy/distance  # 法向量
            
            # 反射速度
            dot_product = self.attacker_marble_vel[0]*nx + self.attacker_marble_vel[1]*ny
            self.attacker_marble_vel[0] = self.attacker_marble_vel[0] - 2 * dot_product * nx
            self.attacker_marble_vel[1] = self.attacker_marble_vel[1] - 2 * dot_product * ny
            
        # 检查是否进入球门
        if (self.attacker_marble_pos[0] >= self.width - self.goal_width and
            self.goal_top <= self.attacker_marble_pos[1] <= self.goal_bottom and
            dist_to_defender > (self.attacker_marble_radius + self.defender_radius)):
            # 得分！
            self.score_a += 1
            rewards['attacker'] += 10.0
            rewards['defender'] -= 10.0
            
            # 重置球位置到初始位置
            self.attacker_marble_pos = [50, self.height // 2]
            self.attacker_marble_vel = [0.0, 0.0]
        
        # 检查与砖块的碰撞
        for i, brick in enumerate(self.bricks):
            dist_to_brick = math.sqrt(
                (self.attacker_marble_pos[0] - brick[0])**2 +
                (self.attacker_marble_pos[1] - brick[1])**2
            )
            
            if dist_to_brick <= self.attacker_marble_radius + 10:  # 砖块半径假设为10
                rewards['attacker'] += 3.0  # 击中砖块得分
                # 可以移除砖块或重置位置
                
        # 基于距离的奖励（鼓励攻击方接近目标，防守方接近球）
        dist_to_goal = abs(self.attacker_marble_pos[0] - self.width)
        rewards['attacker'] += (self.width - dist_to_goal) / self.width * 0.1  # 接近目标奖励
        
        dist_to_ball = math.sqrt(
            (self.defender_pos[0] - self.attacker_marble_pos[0])**2 +
            (self.defender_pos[1] - self.attacker_marble_pos[1])**2
        )
        rewards['defender'] += (1 - dist_to_ball / self.width) * 0.1  # 接近球奖励
        
        return rewards

    def render(self):
        """渲染环境"""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """渲染帧"""
        if self.render_mode != "human":
            return

        canvas = pygame.Surface((self.width, self.height))
        canvas.fill((0, 0, 0))  # 黑色背景
        
        # 绘制球门
        pygame.draw.rect(
            canvas, 
            (0, 255, 0),  # 绿色
            pygame.Rect(self.width - self.goal_width, self.goal_top, self.goal_width, self.goal_bottom - self.goal_top)
        )
        
        # 绘制砖块
        for brick in self.bricks:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),  # 红色
                pygame.Rect(brick[0]-10, brick[1]-10, 20, 20)
            )
        
        # 绘制攻击方弹珠（蓝色）
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # 蓝色
            (int(self.attacker_marble_pos[0]), int(self.attacker_marble_pos[1])),
            self.attacker_marble_radius
        )
        
        # 绘制防守方守门员（黄色）
        pygame.draw.circle(
            canvas,
            (255, 255, 0),  # 黄色
            (int(self.defender_pos[0]), int(self.defender_pos[1])),
            self.defender_radius
        )
        
        # 绘制分数
        score_text = self.font.render(f'Attack Score: {self.score_a}  Defend Score: {self.score_b}', True, (255, 255, 255))
        canvas.blit(score_text, (10, 10))
        
        step_text = self.font.render(f'Step: {self.current_step}/{self.max_steps}', True, (255, 255, 255))
        canvas.blit(step_text, (10, 40))
        
        # 将结果复制到显示窗口
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(60)  # 控制帧率

    def close(self):
        if hasattr(self, 'window'):
            pygame.quit()


def train_marble_combat():
    """
    训练弹珠对抗系统
    """
    print("创建弹珠对抗环境...")
    
    # 创建环境
    env = MarbleCombatEnv(render_mode="human")
    
    print("弹珠对抗环境创建成功！")
    print("攻击方（蓝色弹珠）试图击中砖块或进入球门")
    print("防守方（黄色守门员）试图阻止攻击方得分")
    
    # 重置环境
    obs, info = env.reset()
    
    # 简单的随机策略演示
    for step in range(2000):
        # 攻击方：随机发射力度和角度
        att_action = np.random.uniform(-1, 1, size=(2,))
        # 防守方：随机移动守门员位置
        def_action = np.random.uniform(0, 1, size=(2,))
        
        actions = {'attacker': att_action, 'defender': def_action}
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if step % 100 == 0:
            print(f"Step {step}: A reward = {rewards['attacker']:.2f}, B reward = {rewards['defender']:.2f}, "
                  f"Score A: {info['score_a']}, Score B: {info['score_b']}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()


if __name__ == "__main__":
    train_marble_combat()