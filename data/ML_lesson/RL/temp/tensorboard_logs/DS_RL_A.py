"""
基于强化学习的智能股票交易系统 - Streamlit GUI版
功能：
1. 实时爬取股票数据
2. 设定持股上下限
3. 模型保存、加载与增量训练
4. 新数据回测与可视化
5. 生成次日交易信号和策略
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import akshare as ak
import warnings
import os
import json
from pathlib import Path

warnings.filterwarnings('ignore')


# ==================== 1. 数据爬取模块 ====================
class StockDataFetcher:
    """股票数据爬取器"""

    def __init__(self):
        self.stock_codes = {
            "国新文化": "SH600636",
            "贵州茅台": "SH600519",
            "宁德时代": "SZ300750",
            "招商银行": "SH600036",
            "中国平安": "SH601318"
        }

    def fetch_from_yfinance(self, symbol, period="1y"):
        """从yfinance获取数据"""
        try:
            # 转换中国股票代码格式
            if symbol.startswith("SH"):
                ticker = f"{symbol[2:]}.SS"
            elif symbol.startswith("SZ"):
                ticker = f"{symbol[2:]}.SZ"
            else:
                ticker = symbol

            stock = yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                return None

            # 重命名列以符合我们的格式
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = ['open', 'high', 'low', 'close', 'volume']

            return df

        except Exception as e:
            st.error(f"yfinance数据获取失败: {e}")
            return None

    def fetch_from_akshare(self, symbol, period="1y"):
        """从akshare获取数据"""
        try:
            # 解析股票代码
            if symbol.startswith("SH"):
                stock_code = f"sh{symbol[2:]}"
            elif symbol.startswith("SZ"):
                stock_code = f"sz{symbol[2:]}"
            else:
                stock_code = symbol

            # 计算日期
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")

            # 获取数据
            df = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                                    start_date=start_date, end_date=end_date)

            if df.empty:
                return None

            # 处理数据
            df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            return df

        except Exception as e:
            st.error(f"akshare数据获取失败: {e}")
            return None

    def fetch_stock_data(self, symbol, period="1y", source="yfinance"):
        """获取股票数据"""
        if source == "yfinance":
            df = self.fetch_from_yfinance(symbol, period)
        else:
            df = self.fetch_from_akshare(symbol, period)

        if df is not None and not df.empty:
            # 添加基本技术指标
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['ma60'] = df['close'].rolling(window=60).mean()
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['volatility'] = df['close'].rolling(window=5).std()

            # 计算RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # 计算MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

            # 填充NaN值
            df.fillna(method='bfill', inplace=True)
            df.fillna(method='ffill', inplace=True)

            # 标准化
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].std() > 0:
                    df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()

        return df


# ==================== 2. 数据处理模块 ====================
class StockDataProcessor:
    """股票数据处理器"""

    def __init__(self):
        self.processed_data = None
        self.feature_columns = []

    def prepare_data(self, data, lookback=20):
        """准备训练数据"""
        if data is None or data.empty:
            return None

        # 选择特征列
        norm_cols = [col for col in data.columns if col.endswith('_norm')]
        self.feature_columns = norm_cols

        # 创建特征矩阵
        features = data[norm_cols].values

        # 创建序列数据
        X, y = self.create_sequences(features, lookback)

        return X, y, features

    def create_sequences(self, data, lookback):
        """创建时间序列数据"""
        X, y = [], []
        for i in range(len(data) - lookback - 1):
            X.append(data[i:(i + lookback)])
            y.append(data[i + lookback, 3])  # 假设close_norm在索引3

        return np.array(X), np.array(y)

    def normalize_for_prediction(self, data):
        """为预测准备标准化数据"""
        norm_data = data.copy()
        for col in norm_data.select_dtypes(include=[np.number]).columns:
            if norm_data[col].std() > 0:
                norm_data[col] = (norm_data[col] - norm_data[col].mean()) / norm_data[col].std()
        return norm_data


# ==================== 3. 交易环境模块 ====================
class StockTradingEnv:
    """股票交易环境 - 支持持股上下限"""

    def __init__(self, data, initial_balance=100000, transaction_cost=0.001,
                 max_shares=1000, min_shares=0):
        """
        初始化环境
        data: pandas DataFrame
        max_shares: 最大持股数量
        min_shares: 最小持股数量
        """
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_shares = max_shares
        self.min_shares = min_shares

        # 确定收盘价列
        self.close_col = 'close_norm' if 'close_norm' in data.columns else 'close'

        # 重置环境
        self.reset()

    def reset(self):
        """重置环境"""
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_value_history = []
        self.action_history = []

        self.current_step = 0
        self.max_steps = len(self.data) - 1

        return self._get_state()

    def _get_state(self):
        """获取当前状态"""
        if self.current_step >= len(self.data):
            return np.zeros(len(self.data.columns) + 4)

        # 市场特征
        market_features = self.data.iloc[self.current_step].values

        # 账户特征
        account_features = np.array([
            self.balance / self.initial_balance,
            self.shares_held / self.max_shares if self.max_shares > 0 else 0,
            self._get_portfolio_value() / self.initial_balance,
            self.current_step / self.max_steps if self.max_steps > 0 else 0
        ])

        return np.concatenate([market_features, account_features])

    def _get_portfolio_value(self):
        """计算当前资产总值"""
        current_price = self._get_current_price()
        return self.balance + self.shares_held * current_price

    def _get_current_price(self):
        """获取当前价格"""
        if self.current_step < len(self.data):
            return self.data.iloc[self.current_step][self.close_col]
        return 0

    def step(self, action):
        """执行交易动作"""
        if self.current_step >= self.max_steps:
            done = True
            next_state = self._get_state()
            reward = self._calculate_final_reward()
            return next_state, reward, done, self._get_info()

        current_price = self._get_current_price()
        prev_value = self._get_portfolio_value()

        # 执行交易动作
        if action == 1:  # 买入
            self._execute_buy(current_price)
        elif action == 2:  # 卖出
            self._execute_sell(current_price)

        # 移动到下一步
        self.current_step += 1

        # 计算新状态和奖励
        next_state = self._get_state()
        new_value = self._get_portfolio_value()

        # 计算奖励
        if prev_value > 0:
            reward = (new_value - prev_value) / prev_value * 100
        else:
            reward = 0

        # 添加交易惩罚（避免过度交易）
        if action != 0:  # 非持有动作
            reward -= 0.05  # 交易成本惩罚

        # 检查是否结束
        done = self.current_step >= self.max_steps

        # 记录历史
        self.total_value_history.append(new_value)
        self.action_history.append(action)

        return next_state, reward, done, self._get_info()

    def _execute_buy(self, current_price):
        """执行买入操作"""
        # 计算可买入的最大股数（考虑持股上限）
        available_shares = min(
            self.max_shares - self.shares_held,
            int(self.balance // (current_price * (1 + self.transaction_cost)))
        )

        if available_shares > 0:
            cost = available_shares * current_price * (1 + self.transaction_cost)
            self.balance -= cost
            self.shares_held += available_shares
            self.total_shares_bought += available_shares

    def _execute_sell(self, current_price):
        """执行卖出操作"""
        # 计算可卖出的股数（考虑持股下限）
        available_shares = self.shares_held - self.min_shares

        if available_shares > 0:
            revenue = available_shares * current_price * (1 - self.transaction_cost)
            self.balance += revenue
            self.shares_held -= available_shares
            self.total_shares_sold += available_shares

    def _calculate_final_reward(self):
        """计算最终奖励"""
        final_value = self._get_portfolio_value()
        total_return = (final_value - self.initial_balance) / self.initial_balance

        # 考虑风险调整回报
        if len(self.total_value_history) > 1:
            returns = pd.Series(self.total_value_history).pct_change().dropna()
            if returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                total_return *= (1 + sharpe_ratio)

        return total_return * 100

    def _get_info(self):
        """获取环境信息"""
        return {
            'price': self._get_current_price(),
            'balance': self.balance,
            'shares_held': self.shares_held,
            'total_value': self._get_portfolio_value(),
            'step': self.current_step,
            'total_shares_bought': self.total_shares_bought,
            'total_shares_sold': self.total_shares_sold
        }


# ==================== 4. 神经网络模块 ====================
class DQNNetwork(nn.Module):
    """深度Q网络"""

    def __init__(self, input_size, output_size, hidden_layers=[128, 64, 32]):
        super(DQNNetwork, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ==================== 5. DQN智能体模块 ====================
class DQNAgent:
    """DQN智能体 - 支持增量训练"""

    def __init__(self, state_size, action_size, config=None):
        self.state_size = state_size
        self.action_size = action_size

        # 配置参数
        self.config = config or {
            'gamma': 0.95,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'learning_rate': 0.001,
            'batch_size': 32,
            'memory_size': 2000,
            'target_update_freq': 10
        }

        # 经验回放缓冲区
        self.memory = deque(maxlen=self.config['memory_size'])

        # 创建网络
        self.model = DQNNetwork(state_size, action_size)
        self.target_model = DQNNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()

        # 同步目标网络
        self.update_target_model()

        # 训练统计
        self.training_history = {
            'rewards': [],
            'losses': [],
            'epsilons': []
        }

    def update_target_model(self):
        """更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """选择动作"""
        if training and np.random.rand() <= self.config['epsilon']:
            return random.randrange(self.action_size)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values[0]).item()

    def replay(self):
        """经验回放学习"""
        if len(self.memory) < self.config['batch_size']:
            return 0

        # 随机采样
        batch = random.sample(self.memory, self.config['batch_size'])

        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.LongTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.FloatTensor([exp[4] for exp in batch])

        # 计算当前Q值
        current_q = self.model(states).gather(1, actions.unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config['gamma'] * next_q

        # 计算损失
        loss = self.criterion(current_q.squeeze(), target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # 衰减探索率
        if self.config['epsilon'] > self.config['epsilon_min']:
            self.config['epsilon'] *= self.config['epsilon_decay']

        return loss.item()

    def train(self, env, episodes=100, callback=None):
        """训练智能体"""
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < 200:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)

                self.remember(state, action, reward, next_state, done)
                loss = self.replay()

                state = next_state
                total_reward += reward
                steps += 1

                if loss > 0:
                    self.training_history['losses'].append(loss)

            # 定期更新目标网络
            if episode % self.config['target_update_freq'] == 0:
                self.update_target_model()

            # 记录训练历史
            self.training_history['rewards'].append(total_reward)
            self.training_history['epsilons'].append(self.config['epsilon'])

            # 回调函数
            if callback:
                callback(episode, episodes, total_reward, self.config['epsilon'])

        return self.training_history

    def save(self, filepath):
        """保存模型"""
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'training_history': self.training_history
        }

        torch.save(save_data, filepath)
        st.success(f"模型已保存到: {filepath}")

    def load(self, filepath):
        """加载模型"""
        if not os.path.exists(filepath):
            st.error(f"模型文件不存在: {filepath}")
            return False

        try:
            checkpoint = torch.load(filepath, map_location='cpu')

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.config = checkpoint['config']
            self.state_size = checkpoint['state_size']
            self.action_size = checkpoint['action_size']
            self.training_history = checkpoint.get('training_history', {
                'rewards': [], 'losses': [], 'epsilons': []
            })

            self.update_target_model()
            st.success(f"模型已从 {filepath} 加载")
            return True

        except Exception as e:
            st.error(f"加载模型失败: {e}")
            return False

    def continue_training(self, env, additional_episodes=50, callback=None):
        """继续训练（增量训练）"""
        st.info(f"开始增量训练: {additional_episodes} 回合")
        history = self.train(env, additional_episodes, callback)
        return history


# ==================== 6. 回测与可视化模块 ====================
class Backtester:
    """回测与结果可视化"""

    def __init__(self):
        self.results = {}

    def run_backtest(self, agent, env, data):
        """运行回测"""
        state = env.reset()
        done = False

        portfolio_values = []
        actions = []
        prices = []
        balances = []
        holdings = []

        # 关闭探索
        original_epsilon = agent.config['epsilon']
        agent.config['epsilon'] = 0.0

        step = 0
        while not done and step < len(data):
            action = agent.act(state, training=False)
            state, reward, done, info = env.step(action)

            portfolio_values.append(info['total_value'])
            actions.append(action)
            prices.append(info['price'])
            balances.append(info['balance'])
            holdings.append(info['shares_held'])
            step += 1

        # 恢复探索率
        agent.config['epsilon'] = original_epsilon

        # 计算绩效指标
        initial_value = env.initial_balance
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value * 100

        # 计算年化收益率
        if len(portfolio_values) > 252:
            annual_return = (final_value / initial_value) ** (252 / len(portfolio_values)) - 1
            annual_return *= 100
        else:
            annual_return = total_return

        # 计算最大回撤
        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # 计算夏普比率
        returns = pd.Series(portfolio_values).pct_change().dropna()
        if returns.std() > 0 and len(returns) > 1:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # 计算胜率
        trades = [i for i, a in enumerate(actions) if a != 0]
        if len(trades) > 1:
            profitable_trades = sum(1 for i in range(1, len(trades))
                                    if portfolio_values[trades[i]] > portfolio_values[trades[i - 1]])
            win_rate = profitable_trades / len(trades) * 100
        else:
            win_rate = 0

        # 保存结果
        self.results = {
            'portfolio_values': portfolio_values,
            'actions': actions,
            'prices': prices,
            'balances': balances,
            'holdings': holdings,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': sum(1 for a in actions if a != 0),
            'buy_trades': sum(1 for a in actions if a == 1),
            'sell_trades': sum(1 for a in actions if a == 2)
        }

        return self.results

    def plot_results(self, data):
        """绘制回测结果"""
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))

        # 1. 价格与交易信号
        dates = data.index[:len(self.results['prices'])]

        axes[0].plot(dates, self.results['prices'], label='价格', linewidth=2)

        # 标记买卖点
        buy_indices = [i for i, a in enumerate(self.results['actions']) if a == 1]
        sell_indices = [i for i, a in enumerate(self.results['actions']) if a == 2]

        if buy_indices:
            axes[0].scatter([dates[i] for i in buy_indices],
                            [self.results['prices'][i] for i in buy_indices],
                            color='green', marker='^', s=100, label='买入', zorder=5)

        if sell_indices:
            axes[0].scatter([dates[i] for i in sell_indices],
                            [self.results['prices'][i] for i in sell_indices],
                            color='red', marker='v', s=100, label='卖出', zorder=5)

        axes[0].set_title('价格走势与交易信号')
        axes[0].set_ylabel('价格')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 资产组合价值
        axes[1].plot(dates, self.results['portfolio_values'],
                     label='组合价值', color='blue', linewidth=2)
        axes[1].axhline(y=100000, color='gray', linestyle='--',
                        alpha=0.5, label='初始资金')
        axes[1].set_title('资产组合价值变化')
        axes[1].set_ylabel('价值 (元)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. 持仓变化
        axes[2].bar(dates, self.results['holdings'],
                    color='purple', alpha=0.7, label='持仓数量')
        axes[2].set_title('持仓数量变化')
        axes[2].set_ylabel('持股数')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # 4. 资金余额
        axes[3].plot(dates, self.results['balances'],
                     color='orange', linewidth=2, label='现金余额')
        axes[3].set_title('现金余额变化')
        axes[3].set_xlabel('日期')
        axes[3].set_ylabel('余额 (元)')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_performance_metrics(self):
        """绘制绩效指标"""
        metrics = {
            '总收益率 (%)': self.results['total_return'],
            '年化收益率 (%)': self.results['annual_return'],
            '最大回撤 (%)': self.results['max_drawdown'],
            '夏普比率': self.results['sharpe_ratio'],
            '胜率 (%)': self.results['win_rate'],
            '总交易次数': self.results['total_trades'],
            '买入次数': self.results['buy_trades'],
            '卖出次数': self.results['sell_trades']
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']

        bars = ax.bar(range(len(metrics)), list(metrics.values()),
                      color=colors[:len(metrics)])

        ax.set_xlabel('绩效指标')
        ax.set_ylabel('数值')
        ax.set_title('回测绩效指标汇总')
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(list(metrics.keys()), rotation=45, ha='right')

        # 在柱子上添加数值标签
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + max(metrics.values()) * 0.01,
                    f'{value:.2f}', ha='center', va='bottom')

        plt.tight_layout()
        return fig


# ==================== 7. 交易信号生成模块 ====================
class TradingSignalGenerator:
    """交易信号生成器"""

    def __init__(self, agent, data_processor):
        self.agent = agent
        self.data_processor = data_processor

    def generate_next_day_signal(self, recent_data):
        """生成次日交易信号"""
        if recent_data is None or len(recent_data) < 20:
            return None

        # 准备特征数据
        features = self.data_processor.normalize_for_prediction(recent_data)

        # 获取最后20天的数据作为状态
        if len(features) >= 20:
            state_data = features.iloc[-20:].values
            # 展平为状态向量
            state = state_data.flatten()

            # 添加账户信息（假设空仓）
            account_info = np.array([1.0, 0.0, 0.0, 1.0])  # 全现金，无持仓
            state = np.concatenate([state, account_info])

            # 使用智能体预测动作
            action = self.agent.act(state, training=False)

            # 获取Q值
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.agent.model(state_tensor).numpy()[0]

            # 生成信号详情
            action_names = {0: "持有", 1: "买入", 2: "卖出"}

            signal = {
                'action': action,
                'action_name': action_names[action],
                'confidence': float(np.max(q_values)),
                'q_values': {i: float(q_values[i]) for i in range(3)},
                'recommendation': self._generate_recommendation(action, q_values),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            return signal

        return None

    def _generate_recommendation(self, action, q_values):
        """生成交易建议"""
        if action == 1:  # 买入
            confidence = q_values[1]
            if confidence > 0.8:
                return "强烈建议买入"
            elif confidence > 0.6:
                return "建议买入"
            else:
                return "谨慎买入"

        elif action == 2:  # 卖出
            confidence = q_values[2]
            if confidence > 0.8:
                return "强烈建议卖出"
            elif confidence > 0.6:
                return "建议卖出"
            else:
                return "谨慎卖出"

        else:  # 持有
            confidence = q_values[0]
            if confidence > 0.8:
                return "强烈建议持有"
            elif confidence > 0.6:
                return "建议持有"
            else:
                return "谨慎持有"


# ==================== 8. Streamlit主应用 ====================
def main():
    """Streamlit主应用"""

    st.set_page_config(
        page_title="智能股票交易系统",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 应用标题
    st.title("🤖 基于强化学习的智能股票交易系统")
    st.markdown("---")

    # 初始化会话状态
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'env' not in st.session_state:
        st.session_state.env = None
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None

    # 侧边栏 - 系统配置
    with st.sidebar:
        st.header("⚙️ 系统配置")

        # 股票选择
        fetcher = StockDataFetcher()
        stock_options = list(fetcher.stock_codes.keys())
        selected_stock = st.selectbox("选择股票", stock_options)
        stock_symbol = fetcher.stock_codes[selected_stock]

        # 数据源选择
        data_source = st.radio("数据源", ["yfinance", "akshare"])

        # 持股限制
        st.subheader("📊 持股限制")
        max_shares = st.number_input("最大持股数", min_value=100, max_value=10000,
                                     value=1000, step=100)
        min_shares = st.number_input("最小持股数", min_value=0, max_value=1000,
                                     value=0, step=10)

        # 训练参数
        st.subheader("🎯 训练参数")
        initial_balance = st.number_input("初始资金 (元)", min_value=10000,
                                          max_value=1000000, value=100000, step=10000)
        transaction_cost = st.slider("交易成本 (%)", min_value=0.0, max_value=1.0,
                                     value=0.1, step=0.01) / 100

        # 模型管理
        st.subheader("💾 模型管理")
        model_name = st.text_input("模型名称", value=f"model_{stock_symbol}")

        col1, col2 = st.columns(2)
        with col1:
            save_model = st.button("💾 保存模型")
        with col2:
            load_model = st.button("📂 加载模型")

    # 主界面布局
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📥 数据获取",
        "🎯 模型训练",
        "📊 回测分析",
        "🔮 交易信号",
        "📈 系统监控"
    ])

    # Tab 1: 数据获取
    with tab1:
        st.header("📥 股票数据获取")

        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("数据周期", ["3mo", "6mo", "1y", "2y", "5y"], index=2)

        with col2:
            if st.button("🚀 获取数据", type="primary"):
                with st.spinner("正在获取股票数据..."):
                    fetcher = StockDataFetcher()
                    data = fetcher.fetch_stock_data(stock_symbol, period, data_source)

                    if data is not None:
                        st.session_state.data = data
                        st.success(f"成功获取 {selected_stock} 数据！")

                        # 显示数据摘要
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("数据天数", len(data))
                        with col2:
                            st.metric("最新价格", f"{data['close'].iloc[-1]:.2f}")
                        with col3:
                            price_change = ((data['close'].iloc[-1] - data['close'].iloc[-2]) /
                                            data['close'].iloc[-2] * 100)
                            st.metric("日涨跌幅", f"{price_change:.2f}%")

                        # 显示价格图表
                        st.subheader("📈 价格走势")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        ax.plot(data.index, data['close'], label='收盘价', linewidth=2)
                        ax.plot(data.index, data['ma20'], label='20日均线', linestyle='--')
                        ax.fill_between(data.index, data['close'].min(), data['close'].max(),
                                        alpha=0.1, color='gray')
                        ax.set_xlabel('日期')
                        ax.set_ylabel('价格 (元)')
                        ax.set_title(f'{selected_stock} 价格走势')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

                        # 显示数据预览
                        st.subheader("📋 数据预览")
                        st.dataframe(data.tail(10))
                    else:
                        st.error("获取数据失败，请检查网络连接或股票代码")

        if st.session_state.data is not None:
            st.download_button(
                label="📥 下载数据 (CSV)",
                data=st.session_state.data.to_csv().encode('utf-8'),
                file_name=f"{stock_symbol}_data.csv",
                mime="text/csv"
            )

    # Tab 2: 模型训练
    with tab2:
        st.header("🎯 强化学习模型训练")

        if st.session_state.data is None:
            st.warning("请先获取股票数据！")
        else:
            col1, col2, col3 = st.columns(3)
            with col1:
                episodes = st.number_input("训练回合数", min_value=10, max_value=1000,
                                           value=100, step=10)
            with col2:
                lookback = st.number_input("回看周期", min_value=5, max_value=60,
                                           value=20, step=5)
            with col3:
                hidden_layers = st.text_input("隐藏层结构", value="128,64,32")

            # 训练按钮
            col1, col2 = st.columns(2)
            with col1:
                train_new = st.button("🆕 开始新训练", type="primary")
            with col2:
                continue_train = st.button("➡️ 继续训练")

            # 训练进度
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 训练回调函数
            def training_callback(episode, total_episodes, reward, epsilon):
                progress = (episode + 1) / total_episodes
                progress_bar.progress(progress)
                status_text.text(f"回合 {episode + 1}/{total_episodes} | "
                                 f"奖励: {reward:.2f} | 探索率: {epsilon:.3f}")

            # 新训练
            if train_new:
                with st.spinner("正在初始化并训练模型..."):
                    # 创建环境
                    env = StockTradingEnv(
                        st.session_state.data,
                        initial_balance=initial_balance,
                        transaction_cost=transaction_cost,
                        max_shares=max_shares,
                        min_shares=min_shares
                    )

                    # 创建智能体
                    state_size = len(env.reset())
                    action_size = 3

                    # 解析隐藏层
                    try:
                        hidden_layers_list = [int(x.strip()) for x in hidden_layers.split(',')]
                    except:
                        hidden_layers_list = [128, 64, 32]

                    agent = DQNAgent(state_size, action_size)

                    # 训练
                    history = agent.train(env, episodes, training_callback)

                    # 保存到会话状态
                    st.session_state.agent = agent
                    st.session_state.env = env

                    # 显示训练结果
                    st.success("训练完成！")

                    # 绘制训练历史
                    st.subheader("📊 训练历史")

                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                    # 奖励曲线
                    axes[0, 0].plot(history['rewards'])
                    axes[0, 0].set_title('训练奖励')
                    axes[0, 0].set_xlabel('回合')
                    axes[0, 0].set_ylabel('奖励')
                    axes[0, 0].grid(True, alpha=0.3)

                    # 损失曲线
                    if history['losses']:
                        axes[0, 1].plot(history['losses'])
                        axes[0, 1].set_title('训练损失')
                        axes[0, 1].set_xlabel('训练步数')
                        axes[0, 1].set_ylabel('损失')
                        axes[0, 1].grid(True, alpha=0.3)

                    # 探索率
                    axes[1, 0].plot(history['epsilons'])
                    axes[1, 0].set_title('探索率衰减')
                    axes[1, 0].set_xlabel('回合')
                    axes[1, 0].set_ylabel('探索率')
                    axes[1, 0].grid(True, alpha=0.3)

                    # 移动平均奖励
                    if len(history['rewards']) > 10:
                        moving_avg = pd.Series(history['rewards']).rolling(window=10).mean()
                        axes[1, 1].plot(history['rewards'], alpha=0.3, label='原始奖励')
                        axes[1, 1].plot(moving_avg, linewidth=2, label='10期移动平均')
                        axes[1, 1].set_title('奖励移动平均')
                        axes[1, 1].set_xlabel('回合')
                        axes[1, 1].set_ylabel('奖励')
                        axes[1, 1].legend()
                        axes[1, 1].grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig)

            # 继续训练
            if continue_train and st.session_state.agent is not None:
                with st.spinner("继续训练模型..."):
                    additional_episodes = st.number_input("增加训练回合数",
                                                          min_value=10, max_value=500,
                                                          value=50, step=10)

                    if st.session_state.env is None:
                        env = StockTradingEnv(
                            st.session_state.data,
                            initial_balance=initial_balance,
                            transaction_cost=transaction_cost,
                            max_shares=max_shares,
                            min_shares=min_shares
                        )
                        st.session_state.env = env

                    history = st.session_state.agent.continue_training(
                        st.session_state.env, additional_episodes, training_callback
                    )

                    st.success(f"增量训练完成！共训练 {len(history['rewards'])} 回合")

            # 模型保存
            if save_model and st.session_state.agent is not None:
                filepath = f"{model_name}.pth"
                st.session_state.agent.save(filepath)

            # 模型加载
            if load_model:
                filepath = f"{model_name}.pth"
                agent = DQNAgent(1, 1)  # 临时创建
                if agent.load(filepath):
                    st.session_state.agent = agent

                    # 创建相应的环境
                    if st.session_state.data is not None:
                        env = StockTradingEnv(
                            st.session_state.data,
                            initial_balance=initial_balance,
                            transaction_cost=transaction_cost,
                            max_shares=max_shares,
                            min_shares=min_shares
                        )
                        st.session_state.env = env

    # Tab 3: 回测分析
    with tab3:
        st.header("📊 回测分析")

        if st.session_state.agent is None or st.session_state.data is None:
            st.warning("请先训练或加载模型并获取数据！")
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 运行回测", type="primary"):
                    with st.spinner("正在运行回测..."):
                        backtester = Backtester()
                        results = backtester.run_backtest(
                            st.session_state.agent,
                            st.session_state.env,
                            st.session_state.data
                        )

                        st.session_state.backtest_results = results

                        # 显示关键指标
                        st.subheader("📈 回测结果摘要")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("总收益率", f"{results['total_return']:.2f}%")
                        with col2:
                            st.metric("年化收益率", f"{results['annual_return']:.2f}%")
                        with col3:
                            st.metric("最大回撤", f"{results['max_drawdown']:.2f}%")
                        with col4:
                            st.metric("夏普比率", f"{results['sharpe_ratio']:.2f}")

            with col2:
                if st.button("🔄 使用新数据回测"):
                    # 重新获取最新数据
                    with st.spinner("获取最新数据并回测..."):
                        fetcher = StockDataFetcher()
                        new_data = fetcher.fetch_stock_data(stock_symbol, period, data_source)

                        if new_data is not None:
                            # 创建新环境
                            new_env = StockTradingEnv(
                                new_data,
                                initial_balance=initial_balance,
                                transaction_cost=transaction_cost,
                                max_shares=max_shares,
                                min_shares=min_shares
                            )

                            backtester = Backtester()
                            results = backtester.run_backtest(
                                st.session_state.agent,
                                new_env,
                                new_data
                            )

                            st.session_state.backtest_results = results
                            st.success("新数据回测完成！")

            # 显示回测结果
            if st.session_state.backtest_results is not None:
                results = st.session_state.backtest_results

                # 绘制回测图表
                st.subheader("📊 回测可视化")

                backtester = Backtester()
                backtester.results = results

                fig1 = backtester.plot_results(st.session_state.data)
                st.pyplot(fig1)

                fig2 = backtester.plot_performance_metrics()
                st.pyplot(fig2)

                # 显示详细统计
                st.subheader("📋 详细统计")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**交易统计**")
                    trade_stats = {
                        "总交易次数": results['total_trades'],
                        "买入次数": results['buy_trades'],
                        "卖出次数": results['sell_trades'],
                        "胜率": f"{results['win_rate']:.2f}%"
                    }
                    for key, value in trade_stats.items():
                        st.write(f"- {key}: {value}")

                with col2:
                    st.write("**风险评估**")
                    risk_stats = {
                        "最大回撤": f"{results['max_drawdown']:.2f}%",
                        "夏普比率": f"{results['sharpe_ratio']:.2f}",
                        "总收益率": f"{results['total_return']:.2f}%",
                        "年化收益率": f"{results['annual_return']:.2f}%"
                    }
                    for key, value in risk_stats.items():
                        st.write(f"- {key}: {value}")

                # 导出回测结果
                st.download_button(
                    label="📥 下载回测报告",
                    data=json.dumps(results, indent=2, default=str),
                    file_name=f"{stock_symbol}_backtest_report.json",
                    mime="application/json"
                )

    # Tab 4: 交易信号
    with tab4:
        st.header("🔮 交易信号生成")

        if st.session_state.agent is None:
            st.warning("请先训练或加载模型！")
        else:
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🎯 生成交易信号", type="primary"):
                    with st.spinner("分析市场数据并生成信号..."):
                        # 获取最新数据
                        fetcher = StockDataFetcher()
                        recent_data = fetcher.fetch_stock_data(stock_symbol, "1mo", data_source)

                        if recent_data is not None:
                            # 生成信号
                            processor = StockDataProcessor()
                            generator = TradingSignalGenerator(st.session_state.agent, processor)
                            signal = generator.generate_next_day_signal(recent_data)

                            if signal:
                                st.session_state.last_signal = signal

                                # 显示信号
                                st.subheader("📢 交易信号")

                                # 信号卡片
                                action_colors = {
                                    "买入": "🟢",
                                    "卖出": "🔴",
                                    "持有": "🟡"
                                }

                                action_color = action_colors.get(signal['action_name'], "⚪")

                                st.markdown(f"""
                                <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
                                    <h2 style="color: white;">{action_color} {signal['action_name']}</h2>
                                    <h3 style="color: white;">置信度: {signal['confidence']:.2%}</h3>
                                    <p style="color: white;">{signal['recommendation']}</p>
                                </div>
                                """, unsafe_allow_html=True)

                                # Q值分布
                                st.subheader("🧠 智能体决策分析")

                                q_values = signal['q_values']
                                actions = ["持有", "买入", "卖出"]

                                fig, ax = plt.subplots(figsize=(10, 6))
                                bars = ax.bar(actions, [q_values[i] for i in range(3)],
                                              color=['yellow', 'green', 'red'])
                                ax.set_xlabel('动作')
                                ax.set_ylabel('Q值')
                                ax.set_title('各动作Q值分布')
                                ax.grid(True, alpha=0.3)

                                # 添加数值标签
                                for bar, value in zip(bars, [q_values[i] for i in range(3)]):
                                    height = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                                            f'{value:.4f}', ha='center', va='bottom')

                                st.pyplot(fig)

                                # 交易建议
                                st.subheader("💡 投资建议")

                                if signal['action'] == 1:  # 买入
                                    st.info("""
                                    **买入建议策略:**
                                    1. 分批建仓，避免一次性全仓买入
                                    2. 设置止损位（建议: -5%）
                                    3. 关注关键技术支撑位
                                    4. 建议买入仓位: 20-30%
                                    """)
                                elif signal['action'] == 2:  # 卖出
                                    st.warning("""
                                    **卖出建议策略:**
                                    1. 分批减仓，锁定利润
                                    2. 设置止盈位（建议: +8%）
                                    3. 关注关键技术阻力位
                                    4. 建议卖出仓位: 30-50%
                                    """)
                                else:  # 持有
                                    st.info("""
                                    **持有建议策略:**
                                    1. 继续持有现有仓位
                                    2. 关注市场动态
                                    3. 设置移动止损保护利润
                                    4. 等待更明确信号
                                    """)

                                # 市场分析
                                st.subheader("📊 市场分析")

                                if len(recent_data) > 0:
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        current_price = recent_data['close'].iloc[-1]
                                        st.metric("当前价格", f"{current_price:.2f}")

                                    with col2:
                                        ma20 = recent_data['ma20'].iloc[-1]
                                        ma_diff = (current_price - ma20) / ma20 * 100
                                        st.metric("20日均线", f"{ma20:.2f}",
                                                  f"{ma_diff:.2f}%")

                                    with col3:
                                        rsi = recent_data['rsi'].iloc[-1] if 'rsi' in recent_data.columns else 50
                                        st.metric("RSI指标", f"{rsi:.1f}")

                                st.caption(f"信号生成时间: {signal['timestamp']}")
                            else:
                                st.error("生成交易信号失败，数据不足")
                        else:
                            st.error("获取最新数据失败")

            with col2:
                # 历史信号记录
                st.subheader("📝 历史信号")

                if 'signal_history' not in st.session_state:
                    st.session_state.signal_history = []

                if st.button("📥 保存当前信号"):
                    if 'last_signal' in st.session_state:
                        st.session_state.signal_history.append(st.session_state.last_signal)
                        st.success("信号已保存到历史记录")

                if st.session_state.signal_history:
                    history_df = pd.DataFrame(st.session_state.signal_history)
                    st.dataframe(history_df[['action_name', 'confidence', 'recommendation', 'timestamp']])

                    # 信号统计
                    st.subheader("📈 信号统计")

                    if not history_df.empty:
                        signal_counts = history_df['action_name'].value_counts()

                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.pie(signal_counts.values, labels=signal_counts.index,
                               autopct='%1.1f%%', startangle=90)
                        ax.set_title('交易信号分布')
                        st.pyplot(fig)

    # Tab 5: 系统监控
    with tab5:
        st.header("📈 系统监控")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("模型状态",
                      "✅ 已加载" if st.session_state.agent else "❌ 未加载")

        with col2:
            data_status = "✅ 已加载" if st.session_state.data is not None else "❌ 未加载"
            st.metric("数据状态", data_status)

        with col3:
            if st.session_state.backtest_results:
                perf = st.session_state.backtest_results['total_return']
                st.metric("最新回测收益", f"{perf:.2f}%")
            else:
                st.metric("最新回测收益", "N/A")

        # 系统信息
        st.subheader("🖥️ 系统信息")

        sys_info = {
            "PyTorch版本": torch.__version__,
            "Streamlit版本": st.__version__,
            "Pandas版本": pd.__version__,
            "NumPy版本": np.__version__,
            "当前时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "工作目录": os.getcwd()
        }

        for key, value in sys_info.items():
            st.write(f"**{key}:** {value}")

        # 磁盘空间检查
        st.subheader("💾 存储空间")

        try:
            total, used, free = shutil.disk_usage("/")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("总空间", f"{total // (2 ** 30):.1f} GB")
            with col2:
                st.metric("已使用", f"{used // (2 ** 30):.1f} GB")
            with col3:
                st.metric("可用空间", f"{free // (2 ** 30):.1f} GB")

            # 存储使用进度条
            usage_percent = used / total * 100
            st.progress(usage_percent / 100, text=f"存储使用率: {usage_percent:.1f}%")

        except:
            st.warning("无法获取存储空间信息")

        # 模型文件管理
        st.subheader("📁 模型文件")

        model_files = [f for f in os.listdir('.') if f.endswith('.pth')]

        if model_files:
            for file in model_files:
                file_size = os.path.getsize(file) / 1024  # KB
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.write(file)
                with col2:
                    st.write(f"{file_size:.1f} KB")
                with col3:
                    if st.button("🗑️", key=f"delete_{file}"):
                        os.remove(file)
                        st.rerun()
        else:
            st.info("暂无模型文件")

        # 系统日志
        st.subheader("📋 系统日志")

        log_placeholder = st.empty()

        if st.button("🔄 刷新日志"):
            # 这里可以添加实际的日志读取逻辑
            log_placeholder.info("日志刷新时间: " + datetime.now().strftime("%H:%M:%S"))

        # 快速操作
        st.subheader("⚡ 快速操作")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 重启环境"):
                if st.session_state.data is not None:
                    env = StockTradingEnv(
                        st.session_state.data,
                        initial_balance=initial_balance,
                        transaction_cost=transaction_cost,
                        max_shares=max_shares,
                        min_shares=min_shares
                    )
                    st.session_state.env = env
                    st.success("环境已重启")

        with col2:
            if st.button("🧹 清除缓存"):
                keys = list(st.session_state.keys())
                for key in keys:
                    if key not in ['agent', 'data', 'env', 'backtest_results']:
                        del st.session_state[key]
                st.success("缓存已清除")

        with col3:
            if st.button("📤 导出配置"):
                config = {
                    'stock_symbol': stock_symbol,
                    'initial_balance': initial_balance,
                    'transaction_cost': transaction_cost,
                    'max_shares': max_shares,
                    'min_shares': min_shares,
                    'model_name': model_name,
                    'export_time': datetime.now().isoformat()
                }

                st.download_button(
                    label="下载配置",
                    data=json.dumps(config, indent=2),
                    file_name="trading_system_config.json",
                    mime="application/json"
                )


# ==================== 9. 运行应用 ====================
if __name__ == "__main__":
    # 检查依赖
    try:
        import streamlit as st
        import shutil  # 用于磁盘空间检查
    except ImportError as e:
        print(f"缺少依赖库: {e}")
        print("请安装所需依赖: pip install streamlit torch pandas numpy matplotlib yfinance akshare")
    else:
        main()
