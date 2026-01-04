import numpy as np
import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from marble_combat import MarbleCombatEnv, SingleAgentWrapper
import os


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """
    自定义特征提取器，用于处理单智能体环境的观测
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)
        
        # 获取观测空间的维度
        obs_dim = observation_space.shape[0]
        
        # 创建神经网络层
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.net(observations)


class DualAgentPPO:
    """
    双智能体PPO训练器
    """
    def __init__(self, base_env):
        self.base_env = base_env  # 原始双智能体环境
        
        # 创建包装后的单智能体环境
        self.attacker_env = SingleAgentWrapper(base_env, agent_id='attacker')
        self.defender_env = SingleAgentWrapper(base_env, agent_id='defender')
        
        # 为每个智能体创建独立的PPO模型
        # 攻击方模型
        self.attacker_model = PPO(
            "MlpPolicy", 
            self.attacker_env,
            policy_kwargs=dict(
                features_extractor_class=CustomFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            ),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
        
        # 防守方模型
        self.defender_model = PPO(
            "MlpPolicy", 
            self.defender_env,
            policy_kwargs=dict(
                features_extractor_class=CustomFeatureExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            ),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
        
        # 训练历史记录
        self.training_history = {
            'attacker_rewards': [],
            'defender_rewards': [],
            'scores': []
        }

    def train_step(self, total_timesteps=10000):
        """
        执行一次训练步骤
        """
        print("开始训练攻击方智能体...")
        # 训练攻击方智能体
        self.attacker_model.learn(total_timesteps=total_timesteps)
        
        print("开始训练防守方智能体...")
        # 训练防守方智能体
        self.defender_model.learn(total_timesteps=total_timesteps)

    def alternating_train(self, total_timesteps=100000, cycles=10):
        """
        交替训练两个智能体
        """
        timesteps_per_cycle = total_timesteps // cycles
        
        for cycle in range(cycles):
            print(f"\n=== 训练周期 {cycle + 1}/{cycles} ===")
            
            # 训练攻击方（防守方使用随机策略）
            print("训练攻击方...")
            self.attacker_model.learn(total_timesteps=timesteps_per_cycle)
            
            # 训练防守方（攻击方使用随机策略）
            print("训练防守方...")
            self.defender_model.learn(total_timesteps=timesteps_per_cycle)
            
            # 评估当前性能
            avg_attacker_reward, avg_defender_reward, avg_score_a, avg_score_b = self.evaluate(10)
            
            self.training_history['attacker_rewards'].append(avg_attacker_reward)
            self.training_history['defender_rewards'].append(avg_defender_reward)
            self.training_history['scores'].append((avg_score_a, avg_score_b))
            
            print(f"周期 {cycle + 1} 完成:")
            print(f"  攻击方平均奖励: {avg_attacker_reward:.2f}")
            print(f"  防守方平均奖励: {avg_defender_reward:.2f}")
            print(f"  攻击方平均得分: {avg_score_a:.2f}")
            print(f"  防守方平均得分: {avg_score_b:.2f}")

    def evaluate(self, episodes=10):
        """
        评估两个智能体的性能
        注意：由于使用了包装器，这里需要特殊处理
        """
        # 为简单起见，我们直接使用基础环境进行评估
        total_attacker_reward = 0
        total_defender_reward = 0
        total_score_a = 0
        total_score_b = 0
        
        for ep in range(episodes):
            obs, _ = self.base_env.reset()
            done = False
            
            episode_attacker_reward = 0
            episode_defender_reward = 0
            
            while not done:
                # 使用训练好的模型进行预测
                attacker_obs = obs['attacker']
                defender_obs = obs['defender']
                
                # 获取两个智能体的动作
                attacker_action, _ = self.attacker_model.predict(attacker_obs, deterministic=True)
                defender_action, _ = self.defender_model.predict(defender_obs, deterministic=True)
                
                actions = {
                    'attacker': attacker_action,
                    'defender': defender_action
                }
                
                obs, rewards, terminated, truncated, info = self.base_env.step(actions)
                
                episode_attacker_reward += rewards['attacker']
                episode_defender_reward += rewards['defender']
                
                done = terminated or truncated
            
            total_attacker_reward += episode_attacker_reward
            total_defender_reward += episode_defender_reward
            total_score_a += info['score_a']
            total_score_b += info['score_b']
        
        return (
            total_attacker_reward / episodes,
            total_defender_reward / episodes,
            total_score_a / episodes,
            total_score_b / episodes
        )

    def save_models(self, attacker_path="marble_attacker_model", defender_path="marble_defender_model"):
        """
        保存训练好的模型
        """
        self.attacker_model.save(attacker_path)
        self.defender_model.save(defender_path)
        print(f"模型已保存到 {attacker_path} 和 {defender_path}")

    def load_models(self, attacker_path="marble_attacker_model", defender_path="marble_defender_model"):
        """
        加载训练好的模型
        """
        if os.path.exists(attacker_path + ".zip"):
            self.attacker_model = PPO.load(attacker_path, env=self.attacker_env)
        if os.path.exists(defender_path + ".zip"):
            self.defender_model = PPO.load(defender_path, env=self.defender_env)
        print(f"模型已从 {attacker_path} 和 {defender_path} 加载")


def main():
    """
    主函数：训练和评估双智能体系统
    """
    print("创建弹珠对抗环境...")
    # 创建基础双智能体环境
    base_env = MarbleCombatEnv(render_mode=None)  # 使用无渲染模式进行训练
    
    print("初始化双智能体PPO训练器...")
    dual_agent_trainer = DualAgentPPO(base_env)
    
    print("开始交替训练...")
    dual_agent_trainer.alternating_train(total_timesteps=10000, cycles=3)
    
    print("\n训练完成！评估最终性能...")
    avg_attacker_reward, avg_defender_reward, avg_score_a, avg_score_b = dual_agent_trainer.evaluate(10)
    
    print(f"\n最终评估结果:")
    print(f"攻击方平均奖励: {avg_attacker_reward:.2f}")
    print(f"防守方平均奖励: {avg_defender_reward:.2f}")
    print(f"攻击方平均得分: {avg_score_a:.2f}")
    print(f"防守方平均得分: {avg_score_b:.2f}")
    
    # 保存模型
    dual_agent_trainer.save_models()
    
    print("\n演示训练后的智能体对战...")
    # 切换到渲染模式进行演示
    demo_env = MarbleCombatEnv(render_mode="human")
    
    # 创建模型包装器用于演示
    attacker_model_demo = dual_agent_trainer.attacker_model
    defender_model_demo = dual_agent_trainer.defender_model
    
    obs, _ = demo_env.reset()
    
    for step in range(1000):
        # 使用训练好的模型进行预测
        attacker_obs = obs['attacker']
        defender_obs = obs['defender']
        
        attacker_action, _ = attacker_model_demo.predict(attacker_obs, deterministic=True)
        defender_action, _ = defender_model_demo.predict(defender_obs, deterministic=True)
        
        actions = {
            'attacker': attacker_action,
            'defender': defender_action
        }
        
        obs, rewards, terminated, truncated, info = demo_env.step(actions)
        
        if step % 100 == 0:
            print(f"Step {step}: Score A: {info['score_a']}, Score B: {info['score_b']}")
        
        if terminated or truncated:
            obs, _ = demo_env.reset()
    
    demo_env.close()


if __name__ == "__main__":
    main()