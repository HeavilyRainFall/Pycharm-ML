# 弹珠对抗AI项目

这是一个双智能体强化学习项目，模拟两个智能体在弹珠台环境中进行对抗。

## 项目结构

- `marble_combat.py`: 弹珠对抗环境定义
- `train_marble_agents.py`: 双智能体训练代码
- `requirements.txt`: 项目依赖

## 游戏机制

- **攻击方（蓝色弹珠）**: 试图击中砖块或进入球门得分
- **防守方（黄色守门员）**: 试图阻止攻击方得分

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行项目

```bash
# 运行游戏演示
python marble_combat.py

# 训练智能体
python train_marble_agents.py
```