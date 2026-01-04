"""
弹珠对抗AI主程序
包含游戏演示和训练功能
"""

from marble_combat import MarbleCombatEnv
from train_marble_agents import DualAgentPPO
import pygame
import numpy as np


def run_demo():
    """
    运行弹珠对抗游戏演示
    """
    print("启动弹珠对抗游戏演示...")
    print("游戏说明:")
    print("- 蓝色弹珠：攻击方，尝试击中砖块或进入球门")
    print("- 黄色守门员：防守方，尝试阻止攻击方得分")
    print("- 球门区域为右侧绿色区域")
    print("- 红色方块为可击中的砖块")
    print("\n按 Q 键退出游戏")
    
    # 创建带渲染的环境
    env = MarbleCombatEnv(render_mode="human")
    obs, info = env.reset()
    
    clock = pygame.time.Clock()
    
    running = True
    step = 0
    
    while running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
        
        # 简单的随机策略
        att_action = np.random.uniform(-0.5, 0.5, size=(2,))  # 攻击方动作
        def_action = np.random.uniform(0, 1, size=(2,))  # 防守方动作
        
        actions = {
            'attacker': att_action,
            'defender': def_action
        }
        
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        if step % 100 == 0:
            print(f"Step {step}: Score A: {info['score_a']}, Score B: {info['score_b']}")
        
        if terminated or truncated:
            obs, info = env.reset()
        
        step += 1
        clock.tick(60)  # 限制帧率为60 FPS
    
    env.close()
    print("游戏演示结束")


def train_agents():
    """
    训练双智能体
    """
    print("开始训练双智能体...")
    
    # 创建无渲染环境用于训练
    env = MarbleCombatEnv(render_mode=None)
    
    # 创建训练器
    trainer = DualAgentPPO(env)
    
    # 开始训练
    trainer.alternating_train(total_timesteps=10000, cycles=3)
    
    # 评估
    avg_attacker_reward, avg_defender_reward, avg_score_a, avg_score_b = trainer.evaluate(10)
    
    print(f"\n训练结果:")
    print(f"攻击方平均奖励: {avg_attacker_reward:.2f}")
    print(f"防守方平均奖励: {avg_defender_reward:.2f}")
    print(f"攻击方平均得分: {avg_score_a:.2f}")
    print(f"防守方平均得分: {avg_score_b:.2f}")
    
    # 保存模型
    trainer.save_models()
    
    print("训练完成！")


def main():
    """
    主函数
    """
    print("欢迎使用弹珠对抗AI系统!")
    print("1. 运行游戏演示")
    print("2. 训练智能体")
    print("3. 退出")
    
    while True:
        choice = input("\n请选择 (1-3): ").strip()
        
        if choice == '1':
            run_demo()
        elif choice == '2':
            train_agents()
        elif choice == '3':
            print("谢谢使用！")
            break
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()