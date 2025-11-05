import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, MultiHeadAttention, Dropout, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
import akshare as ak
import warnings
import pickle
import os
from datetime import datetime, timedelta
import json
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12  # 设置字体大小

warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="智能股票预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 配置akshare的请求会话，添加重试机制和延迟
def create_session():
    session = requests.Session()

    # 设置重试策略
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        backoff_factor=1
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # 设置随机User-Agent
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
    ]

    session.headers.update({
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    })

    return session


# 全局会话对象
if 'ak_session' not in st.session_state:
    st.session_state.ak_session = create_session()


class TrainingProgressCallback(Callback):
    """自定义回调函数用于显示训练进度"""

    def __init__(self, progress_bar, status_text, epochs):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(progress)
        self.status_text.text(
            f"训练进度: {epoch + 1}/{self.epochs} - 损失: {logs['loss']:.4f}, 验证损失: {logs['val_loss']:.4f}")


class StockPredictor:
    def __init__(self, time_window=30):
        self.time_window = time_window
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.features = ["开盘", "最高", "最低", "收盘", "成交量"]
        self.close_idx = self.features.index("收盘")
        self.is_trained = False
        self.model_type = "LSTM+Attention"
        self.data_cache_dir = "stock_data_cache"

    def get_stock_data(self, stock_code, start_date="20200101", end_date=None):
        """获取股票数据 - 带有缓存和更新机制"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        # 创建缓存目录
        os.makedirs(self.data_cache_dir, exist_ok=True)

        # 缓存文件路径
        cache_file = os.path.join(self.data_cache_dir, f"{stock_code}.pkl")

        # 尝试从缓存加载数据
        cached_data = None
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                st.info(f"从缓存加载股票 {stock_code} 数据")
            except Exception as e:
                st.warning(f"加载缓存数据失败: {e}")

        # 如果有缓存数据，检查是否需要更新
        if cached_data is not None and not cached_data.empty:
            # 获取缓存数据的最后日期
            last_cached_date = cached_data.index.max().strftime("%Y%m%d")

            # 如果缓存数据已经包含所需的所有日期，直接返回
            if last_cached_date >= end_date:
                return cached_data, f"从缓存获取股票 {stock_code} 数据，共 {len(cached_data)} 行"

            # 否则，只下载新的数据
            start_date = (pd.to_datetime(last_cached_date) + timedelta(days=1)).strftime("%Y%m%d")
            st.info(f"缓存数据需要更新，从 {start_date} 开始下载新数据")

        try:
            # 添加随机延迟，避免请求过于频繁
            time.sleep(random.uniform(1, 3))

            # 使用带有重试机制的akshare
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="hfq"  # 使用后复权数据，便于长期分析
            )

            if df is not None and len(df) > 0:
                # 数据预处理
                df["日期"] = pd.to_datetime(df["日期"])
                df = df.sort_values("日期")
                df.set_index("日期", inplace=True)

                # 选择特征
                available_features = [col for col in self.features if col in df.columns]
                new_data = df[available_features].dropna()

                # 计算技术指标
                new_data = self.add_technical_indicators(new_data)

                # 合并缓存数据和新数据
                if cached_data is not None and not cached_data.empty:
                    # 确保没有重复数据
                    combined_data = pd.concat([cached_data, new_data]).drop_duplicates()
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    combined_data = combined_data.sort_index()
                else:
                    combined_data = new_data

                # 保存到缓存
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(combined_data, f)
                    st.success(f"股票 {stock_code} 数据已缓存")
                except Exception as e:
                    st.warning(f"保存缓存数据失败: {e}")

                return combined_data, f"成功获取股票 {stock_code} 数据，共 {len(combined_data)} 行"
            else:
                # 如果没有新数据，返回缓存数据
                if cached_data is not None and not cached_data.empty:
                    return cached_data, f"无新数据，使用缓存数据，共 {len(cached_data)} 行"
                else:
                    return None, f"股票 {stock_code} 返回空数据"

        except Exception as e:
            # 如果获取真实数据失败，使用缓存数据或模拟数据
            if cached_data is not None and not cached_data.empty:
                st.warning(f"获取股票 {stock_code} 数据失败: {e}，使用缓存数据")
                return cached_data, f"使用缓存数据，共 {len(cached_data)} 行"
            else:
                st.warning(f"获取股票 {stock_code} 数据失败: {e}，使用模拟数据")
                return self.create_simulated_data(stock_code)

    def add_technical_indicators(self, data):
        """添加技术指标"""
        try:
            # 移动平均线
            data['MA5'] = data['收盘'].rolling(window=5).mean()
            data['MA10'] = data['收盘'].rolling(window=10).mean()
            data['MA20'] = data['收盘'].rolling(window=20).mean()

            # 手动计算RSI
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi

            data['RSI'] = calculate_rsi(data['收盘'])

            # 手动计算MACD
            def calculate_macd(prices, fast=12, slow=26, signal=9):
                ema_fast = prices.ewm(span=fast).mean()
                ema_slow = prices.ewm(span=slow).mean()
                macd = ema_fast - ema_slow
                macd_signal = macd.ewm(span=signal).mean()
                macd_histogram = macd - macd_signal
                return macd, macd_signal, macd_histogram

            macd, macd_signal, macd_histogram = calculate_macd(data['收盘'])
            data['MACD'] = macd
            data['MACD_Signal'] = macd_signal
            data['MACD_Histogram'] = macd_histogram

            # 布林带
            def calculate_bollinger_bands(prices, window=20, num_std=2):
                rolling_mean = prices.rolling(window=window).mean()
                rolling_std = prices.rolling(window=window).std()
                upper_band = rolling_mean + (rolling_std * num_std)
                lower_band = rolling_mean - (rolling_std * num_std)
                return upper_band, rolling_mean, lower_band

            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(data['收盘'])
            data['BB_Upper'] = bb_upper
            data['BB_Middle'] = bb_middle
            data['BB_Lower'] = bb_lower

            # 删除NaN值
            data = data.dropna()

        except Exception as e:
            st.warning(f"计算技术指标时出错: {e}")

        return data

    def create_simulated_data(self, stock_name="模拟股票", start_date="2020-01-01", end_date=None):
        """创建模拟数据用于测试"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(42)

        n = len(dates)
        trend = np.linspace(100, 200, n)
        noise = np.cumsum(np.random.randn(n) * 0.8)
        price = trend + noise

        df = pd.DataFrame({
            '开盘': price + np.random.randn(n) * 2,
            '最高': price + np.abs(np.random.randn(n)) * 3 + 2,
            '最低': price - np.abs(np.random.randn(n)) * 3 - 2,
            '收盘': price,
            '成交量': np.random.randint(1000000, 20000000, n) + np.cumsum(np.random.randn(n) * 1000000).astype(int)
        }, index=dates)

        # 为模拟数据添加技术指标
        df = self.add_technical_indicators(df)

        return df, f"创建 {stock_name} 模拟数据，共 {len(df)} 行"

    def prepare_data(self, data):
        """准备训练数据"""
        # 使用更多特征进行训练
        training_features = ["开盘", "最高", "最低", "收盘", "成交量", "MA5", "MA10", "MA20", "RSI", "MACD"]
        available_features = [col for col in training_features if col in data.columns]

        if len(available_features) < 5:  # 如果没有技术指标，使用基本特征
            available_features = self.features

        scaled_data = self.scaler.fit_transform(data[available_features])

        def create_sequences(data, time_window, target_idx):
            X, y = [], []
            for i in range(time_window, len(data)):
                X.append(data[i - time_window:i])
                y.append(data[i, target_idx])
            return np.array(X), np.array(y)

        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # 找到收盘价在特征中的索引
        close_idx = available_features.index("收盘") if "收盘" in available_features else 3

        train_X, train_y = create_sequences(train_data, self.time_window, close_idx)
        test_X, test_y = create_sequences(test_data, self.time_window, close_idx)

        return train_X, train_y, test_X, test_y, train_size, available_features

    def build_model(self, input_shape, model_type="LSTM+Attention", lstm_units=64, gru_units=64,
                    attention_heads=4, dense_units=32, dropout_rate=0.2):
        """构建模型"""
        inputs = Input(shape=input_shape)

        if model_type == "LSTM+Attention":
            lstm_out = LSTM(lstm_units, return_sequences=True, activation='tanh')(inputs)
            lstm_out = Dropout(dropout_rate)(lstm_out)
            attention_out = MultiHeadAttention(num_heads=attention_heads, key_dim=32)(lstm_out, lstm_out)
            combined = lstm_out + attention_out
            last_step = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(combined)

        elif model_type == "GRU":
            gru_out = GRU(gru_units, return_sequences=True, activation='tanh')(inputs)
            gru_out = Dropout(dropout_rate)(gru_out)
            last_step = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(gru_out)

        elif model_type == "CNN+LSTM":
            # CNN部分
            conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
            pool1 = MaxPooling1D(pool_size=2)(conv1)
            conv2 = Conv1D(filters=32, kernel_size=3, activation='relu')(pool1)
            pool2 = MaxPooling1D(pool_size=2)(conv2)
            # LSTM部分
            lstm_out = LSTM(lstm_units, return_sequences=False)(pool2)
            last_step = Dropout(dropout_rate)(lstm_out)

        elif model_type == "Simple LSTM":
            lstm_out = LSTM(lstm_units, return_sequences=False)(inputs)
            last_step = Dropout(dropout_rate)(lstm_out)

        else:  # 默认LSTM+Attention
            lstm_out = LSTM(lstm_units, return_sequences=True, activation='tanh')(inputs)
            lstm_out = Dropout(dropout_rate)(lstm_out)
            attention_out = MultiHeadAttention(num_heads=attention_heads, key_dim=32)(lstm_out, lstm_out)
            combined = lstm_out + attention_out
            last_step = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(combined)

        output = Dense(dense_units, activation='relu')(last_step)
        output = Dropout(dropout_rate)(output)
        output = Dense(1, activation='linear')(output)

        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )

        self.model_type = model_type
        return model

    def train_model(self, stock_codes, epochs=50, use_simulated=False, model_type="LSTM+Attention",
                    lstm_units=64, gru_units=64, attention_heads=4, dense_units=32, dropout_rate=0.2,
                    progress_bar=None, status_text=None):
        """训练模型"""
        all_train_X, all_train_y = [], []
        all_test_X, all_test_y = [], []
        stock_data_info = []

        if not isinstance(stock_codes, list):
            stock_codes = [stock_codes]

        for stock_code in stock_codes:
            if use_simulated:
                data, info = self.create_simulated_data(stock_code)
            else:
                data, info = self.get_stock_data(stock_code)
                if data is None:
                    data, info = self.create_simulated_data(stock_code)

            stock_data_info.append(f"{stock_code}: {info}")

            train_X, train_y, test_X, test_y, _, features_used = self.prepare_data(data)
            self.features = features_used  # 更新使用的特征列表
            self.close_idx = features_used.index("收盘") if "收盘" in features_used else 3

            all_train_X.append(train_X)
            all_train_y.append(train_y)
            all_test_X.append(test_X)
            all_test_y.append(test_y)

        # 合并数据
        train_X = np.vstack(all_train_X)
        train_y = np.hstack(all_train_y)
        test_X = np.vstack(all_test_X)
        test_y = np.hstack(all_test_y)

        # 构建模型
        self.model = self.build_model(
            (self.time_window, len(self.features)),
            model_type=model_type,
            lstm_units=lstm_units,
            gru_units=gru_units,
            attention_heads=attention_heads,
            dense_units=dense_units,
            dropout_rate=dropout_rate
        )

        # 训练模型
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        callbacks = [early_stopping]
        if progress_bar and status_text:
            callbacks.append(TrainingProgressCallback(progress_bar, status_text, epochs))

        history = self.model.fit(
            train_X, train_y,
            epochs=epochs,
            batch_size=32,
            validation_data=(test_X, test_y),
            verbose=0,
            callbacks=callbacks
        )

        self.is_trained = True

        return history, train_X, train_y, test_X, test_y, stock_data_info

    def predict(self, data):
        """对给定数据进行预测"""
        if not self.is_trained:
            return None, "模型未训练"

        # 使用与训练相同的特征
        available_features = [col for col in self.features if col in data.columns]
        if len(available_features) < len(self.features):
            st.warning(f"数据缺少某些特征，使用可用特征: {available_features}")

        data_to_use = data[available_features]
        scaled_data = self.scaler.transform(data_to_use)

        X, y = [], []
        close_idx = available_features.index("收盘") if "收盘" in available_features else 3

        for i in range(self.time_window, len(scaled_data)):
            X.append(scaled_data[i - self.time_window:i])
            y.append(scaled_data[i, close_idx])

        X, y = np.array(X), np.array(y)

        predictions = self.model.predict(X, verbose=0)

        # 反归一化
        predictions_inv = self.inverse_transform_pred(predictions, available_features, close_idx)
        y_inv = self.inverse_transform_pred(y.reshape(-1, 1), available_features, close_idx)

        return predictions_inv, y_inv

    def predict_future(self, data, days=30):
        """预测未来价格走势"""
        if not self.is_trained:
            return None, "模型未训练"

        # 使用与训练相同的特征
        available_features = [col for col in self.features if col in data.columns]
        close_idx = available_features.index("收盘") if "收盘" in available_features else 3

        last_sequence = data[available_features].tail(self.time_window)
        last_sequence_scaled = self.scaler.transform(last_sequence)

        future_predictions = []
        current_sequence = last_sequence_scaled.copy()

        for _ in range(days):
            next_pred = self.model.predict(current_sequence.reshape(1, self.time_window, len(available_features)),
                                           verbose=0)
            future_predictions.append(next_pred[0, 0])

            new_day = current_sequence[-1].copy()
            new_day[close_idx] = next_pred[0, 0]
            current_sequence = np.vstack([current_sequence[1:], new_day])

        future_predictions = self.inverse_transform_pred(np.array(future_predictions).reshape(-1, 1),
                                                         available_features, close_idx)

        return future_predictions

    def inverse_transform_pred(self, y_pred, features, close_idx):
        """反归一化预测结果"""
        y_reshaped = np.zeros(shape=(len(y_pred), len(features)))
        y_reshaped[:, close_idx] = y_pred.flatten()
        return self.scaler.inverse_transform(y_reshaped)[:, close_idx]

    def evaluate_model(self, true_values, predictions):
        """评估模型性能"""
        mae = mean_absolute_error(true_values, predictions)
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_values, predictions)

        # 计算方向准确率
        if len(true_values) > 1:
            true_direction = np.diff(true_values) > 0
            pred_direction = np.diff(predictions) > 0
            direction_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            direction_accuracy = 0

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Direction_Accuracy': direction_accuracy
        }

    def backtest_strategy(self, data, predictions, strategy_type="simple",
                          initial_capital=100000, transaction_cost=0.001,
                          rsi_oversold=30, rsi_overbought=70,
                          ma_short=5, ma_long=20, stop_loss=0.05, take_profit=0.1):
        """
        回测量化策略
        支持多种策略类型
        """
        if len(data) != len(predictions) + self.time_window:
            return None, "数据长度不匹配"

        # 获取实际价格（与预测对应的部分）
        actual_prices = data['收盘'].values[self.time_window:]
        actual_dates = data.index[self.time_window:]

        # 计算技术指标
        data_with_indicators = self.add_technical_indicators(data)

        # 初始化变量
        capital = initial_capital
        position = 0  # 0表示空仓，1表示满仓
        shares = 0
        entry_price = 0
        trades = []
        portfolio_values = []

        # 初始化技术指标变量
        rsi = 50
        ma_short_val = actual_prices[0] if len(actual_prices) > 0 else 0
        ma_long_val = actual_prices[0] if len(actual_prices) > 0 else 0

        for i in range(1, len(predictions)):
            if i >= len(actual_prices):
                break

            current_price = actual_prices[i]
            current_date = actual_dates[i]

            # 获取技术指标 - 添加边界检查
            try:
                if 'RSI' in data_with_indicators.columns and self.time_window + i < len(data_with_indicators):
                    rsi = data_with_indicators['RSI'].iloc[self.time_window + i]
                else:
                    rsi = 50

                if f'MA{ma_short}' in data_with_indicators.columns and self.time_window + i < len(data_with_indicators):
                    ma_short_val = data_with_indicators[f'MA{ma_short}'].iloc[self.time_window + i]
                else:
                    ma_short_val = current_price

                if f'MA{ma_long}' in data_with_indicators.columns and self.time_window + i < len(data_with_indicators):
                    ma_long_val = data_with_indicators[f'MA{ma_long}'].iloc[self.time_window + i]
                else:
                    ma_long_val = current_price
            except (IndexError, KeyError):
                # 如果索引超出范围，使用默认值
                rsi = 50
                ma_short_val = current_price
                ma_long_val = current_price

            # 预测明日涨跌
            predicted_tomorrow = predictions[i]
            predicted_today = predictions[i - 1]
            predicted_change = (predicted_tomorrow - predicted_today) / predicted_today if predicted_today != 0 else 0

            # 策略决策
            trade_signal = 0  # 0: 无信号, 1: 买入, -1: 卖出

            if strategy_type == "simple":
                # 简单策略：基于预测方向
                if predicted_tomorrow > predicted_today and position == 0:
                    trade_signal = 1
                elif predicted_tomorrow < predicted_today and position == 1:
                    trade_signal = -1

            elif strategy_type == "rsi_based":
                # RSI策略：结合RSI超买超卖
                if rsi < rsi_oversold and predicted_change > 0 and position == 0:
                    trade_signal = 1
                elif rsi > rsi_overbought and predicted_change < 0 and position == 1:
                    trade_signal = -1

            elif strategy_type == "ma_crossover":
                # 移动平均线交叉策略
                ma_signal = 1 if ma_short_val > ma_long_val else -1
                if ma_signal > 0 and predicted_change > 0 and position == 0:
                    trade_signal = 1
                elif ma_signal < 0 and predicted_change < 0 and position == 1:
                    trade_signal = -1

            elif strategy_type == "combined":
                # 综合策略：结合多种指标
                ma_signal = 1 if ma_short_val > ma_long_val else -1
                rsi_signal = 1 if rsi < rsi_oversold else (-1 if rsi > rsi_overbought else 0)

                buy_condition = (predicted_change > 0.01 and
                                 (ma_signal > 0 or rsi_signal > 0) and
                                 position == 0)

                sell_condition = (predicted_change < -0.01 and
                                  (ma_signal < 0 or rsi_signal < 0) and
                                  position == 1)

                if buy_condition:
                    trade_signal = 1
                elif sell_condition:
                    trade_signal = -1

            elif strategy_type == "momentum":
                # 动量策略：基于价格动量
                if i >= 5:  # 确保有足够的历史数据
                    price_momentum = (current_price - actual_prices[i - 5]) / actual_prices[i - 5] if actual_prices[
                                                                                                          i - 5] != 0 else 0
                else:
                    price_momentum = 0

                if price_momentum > 0.02 and predicted_change > 0.01 and position == 0:
                    trade_signal = 1
                elif price_momentum < -0.02 and predicted_change < -0.01 and position == 1:
                    trade_signal = -1

            # 止损止盈检查
            if position == 1:
                profit_pct = (current_price - entry_price) / entry_price if entry_price != 0 else 0
                if profit_pct <= -stop_loss or profit_pct >= take_profit:
                    trade_signal = -1  # 强制卖出

            # 执行交易
            if trade_signal == 1 and position == 0:  # 买入
                shares = capital / current_price
                capital = 0
                position = 1
                entry_price = current_price
                trades.append(('BUY', current_date, current_price, f"预测涨幅: {predicted_change:.2%}"))

            elif trade_signal == -1 and position == 1:  # 卖出
                capital = shares * current_price * (1 - transaction_cost)
                shares = 0
                position = 0
                profit = (current_price - entry_price) / entry_price if entry_price != 0 else 0
                trades.append(('SELL', current_date, current_price, f"盈亏: {profit:.2%}"))

            # 计算当前投资组合价值
            if position == 1:
                portfolio_value = shares * current_price
            else:
                portfolio_value = capital

            portfolio_values.append(portfolio_value)

        # 计算最终收益
        if position == 1:  # 如果最后还持有股票，按最后价格卖出
            final_value = shares * actual_prices[-1] * (1 - transaction_cost)
            profit = (actual_prices[-1] - entry_price) / entry_price if entry_price != 0 else 0
            trades.append(('SELL', actual_dates[-1], actual_prices[-1], f"最终盈亏: {profit:.2%}"))
        else:
            final_value = capital

        total_return = (final_value - initial_capital) / initial_capital * 100
        buy_hold_return = (actual_prices[-1] - actual_prices[0]) / actual_prices[0] * 100 if actual_prices[
                                                                                                 0] != 0 else 0

        # 计算最大回撤
        if len(portfolio_values) > 0:
            portfolio_array = np.array(portfolio_values)
            peak = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array - peak) / peak * 100
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        else:
            max_drawdown = 0

        # 计算夏普比率（简化版）
        if len(portfolio_values) > 1:
            portfolio_array = np.array(portfolio_values)
            returns = np.diff(portfolio_array) / portfolio_array[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0

        # 计算年化收益率
        total_days = len(portfolio_values)
        if total_days > 0 and initial_capital > 0:
            annual_return = (final_value / initial_capital) ** (365 / total_days) - 1
        else:
            annual_return = 0

        # 计算胜率
        winning_trades = 0
        total_trades = len([t for t in trades if t[0] == 'SELL'])

        for i in range(1, len(trades)):
            if trades[i][0] == 'SELL':
                # 查找对应的买入交易
                for j in range(i - 1, -1, -1):
                    if trades[j][0] == 'BUY':
                        buy_price = trades[j][2]
                        sell_price = trades[i][2]
                        if sell_price > buy_price:
                            winning_trades += 1
                        break

        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return * 100,
            'buy_hold_return': buy_hold_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'trades': trades,
            'portfolio_values': portfolio_values,
            'dates': actual_dates[1:min(len(predictions), len(actual_dates))],
            'strategy_type': strategy_type
        }, "回测完成"

    def save_model(self, filepath):
        """保存模型和配置"""
        if self.model is None:
            return False, "没有训练好的模型可保存"

        # 创建目录
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        # 保存模型 - 使用SavedModel格式避免自定义层问题
        model_path = filepath if filepath.endswith('.h5') else filepath + '.h5'

        # 使用tf.saved_model.save保存整个模型
        saved_model_dir = model_path.replace('.h5', '')
        tf.saved_model.save(self.model, saved_model_dir)

        # 保存scaler和其他配置
        config = {
            'time_window': self.time_window,
            'features': self.features,
            'close_idx': self.close_idx,
            'is_trained': self.is_trained,
            'model_type': self.model_type,
            'scaler_params': {
                'min_': self.scaler.min_,
                'scale_': self.scaler.scale_,
                'data_min_': self.scaler.data_min_,
                'data_max_': self.scaler.data_max_,
                'data_range_': self.scaler.data_range_
            }
        }

        config_path = filepath.replace('.h5', '_config.pkl') if filepath.endswith('.h5') else filepath + '_config.pkl'
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)

        return True, f"模型已保存到 {saved_model_dir}"

    def load_model(self, filepath):
        """加载模型和配置"""
        # 处理文件路径
        if filepath.endswith('.h5'):
            saved_model_dir = filepath.replace('.h5', '')
            config_path = filepath.replace('.h5', '_config.pkl')
        else:
            saved_model_dir = filepath
            config_path = filepath + '_config.pkl'

        # 检查文件是否存在
        if not os.path.exists(saved_model_dir):
            return False, f"模型文件夹不存在: {saved_model_dir}"

        if not os.path.exists(config_path):
            return False, f"配置文件不存在: {config_path}"

        try:
            # 加载模型 - 使用tf.saved_model.load
            self.model = tf.saved_model.load(saved_model_dir)

            # 获取callable的signature用于预测
            if hasattr(self.model, 'signatures') and 'serving_default' in self.model.signatures:
                self.model = self.model.signatures['serving_default']
            else:
                # 如果无法获取serving_default，尝试直接调用
                self.model = self.model

            # 加载配置
            with open(config_path, 'rb') as f:
                config = pickle.load(f)

            self.time_window = config['time_window']
            self.features = config['features']
            self.close_idx = config['close_idx']
            self.is_trained = config['is_trained']
            self.model_type = config.get('model_type', 'LSTM+Attention')

            # 恢复scaler
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.min_ = config['scaler_params']['min_']
            self.scaler.scale_ = config['scaler_params']['scale_']
            self.scaler.data_min_ = config['scaler_params']['data_min_']
            self.scaler.data_max_ = config['scaler_params']['data_max_']
            self.scaler.data_range_ = config['scaler_params']['data_range_']

            return True, f"模型已从 {saved_model_dir} 加载"

        except Exception as e:
            return False, f"加载模型时出错: {e}"


# 初始化预测器
if 'predictor' not in st.session_state:
    st.session_state.predictor = StockPredictor()
    st.session_state.model_trained = False
    st.session_state.training_history = None

# 界面设计
st.title("📈 智能股票预测系统")
st.markdown("使用深度学习模型进行多股票训练、预测和量化回测")

# 侧边栏
st.sidebar.header("模型配置")

# 时间窗口设置
time_window = st.sidebar.slider("时间窗口大小", min_value=10, max_value=60, value=30, step=5)
st.session_state.predictor.time_window = time_window

# 主界面选项卡
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 模型训练", "🔍 股票验证", "🔮 未来预测", "📊 量化回测", "💾 模型管理"])

with tab1:
    st.header("模型训练")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("训练参数")

        # 股票代码输入
        train_stocks = st.text_area(
            "训练股票代码（用逗号分隔）",
            value="600519,000858,300750",
            help="例如：600519,000858,300750"
        )

        # 模型选择
        model_type = st.selectbox(
            "选择模型类型",
            ["LSTM+Attention", "GRU", "CNN+LSTM", "Simple LSTM"],
            help="选择不同的神经网络架构"
        )

        # 模型参数
        col1a, col1b = st.columns(2)
        with col1a:
            lstm_units = st.slider("LSTM单元数", min_value=16, max_value=256, value=64, step=16)
            gru_units = st.slider("GRU单元数", min_value=16, max_value=256, value=64, step=16)
            attention_heads = st.slider("注意力头数", min_value=2, max_value=8, value=4, step=1)

        with col1b:
            dense_units = st.slider("全连接层单元数", min_value=16, max_value=128, value=32, step=8)
            dropout_rate = st.slider("Dropout比率", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            epochs = st.number_input("训练轮数 (Epochs)", min_value=10, max_value=500, value=50)

        use_simulated = st.checkbox("使用模拟数据（当真实数据不可用时）", value=True)

    with col2:
        st.subheader("训练状态")
        if st.session_state.model_trained:
            st.success("✅ 模型已训练完成")
            if st.session_state.training_history:
                st.write(f"模型类型: {st.session_state.predictor.model_type}")
                st.write(f"最终训练损失: {st.session_state.training_history.history['loss'][-1]:.4f}")
                st.write(f"最终验证损失: {st.session_state.training_history.history['val_loss'][-1]:.4f}")
        else:
            st.warning("⏳ 模型未训练")

        # 训练进度区域
        progress_bar = st.progress(0)
        status_text = st.empty()

    if st.button("开始训练模型", type="primary"):
        if train_stocks.strip():
            stock_list = [s.strip() for s in train_stocks.split(',') if s.strip()]

            st.info(f"开始训练模型，使用 {len(stock_list)} 只股票数据")

            with st.spinner("正在训练模型，请稍候..."):
                history, train_X, train_y, test_X, test_y, stock_info = st.session_state.predictor.train_model(
                    stock_list,
                    epochs=epochs,
                    use_simulated=use_simulated,
                    model_type=model_type,
                    lstm_units=lstm_units,
                    gru_units=gru_units,
                    attention_heads=attention_heads,
                    dense_units=dense_units,
                    dropout_rate=dropout_rate,
                    progress_bar=progress_bar,
                    status_text=status_text
                )

                st.session_state.training_history = history
                st.session_state.model_trained = True

                # 显示训练结果
                st.success("模型训练完成！")

                # 显示股票数据信息
                st.subheader("训练数据信息")
                for info in stock_info:
                    st.write(f"- {info}")

                # 绘制训练历史
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                ax1.plot(history.history['loss'], label='训练损失')
                ax1.plot(history.history['val_loss'], label='验证损失')
                ax1.set_title('模型损失')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                ax2.plot(history.history['mae'], label='训练MAE')
                ax2.plot(history.history['val_mae'], label='验证MAE')
                ax2.set_title('模型MAE')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MAE')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                st.pyplot(fig)

        else:
            st.error("请输入至少一个股票代码")

with tab2:
    st.header("股票验证")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("验证参数")
        validate_stock = st.text_input("验证股票代码", value="000001")
        use_current_model = st.checkbox("使用当前训练好的模型", value=True)

    with col2:
        st.subheader("验证说明")
        st.info("""
        使用训练好的模型对新的股票数据进行验证，
        评估模型在未见过的股票上的表现。
        """)

    if st.button("开始验证", type="primary"):
        if validate_stock.strip() and st.session_state.model_trained:
            with st.spinner("正在验证模型..."):
                # 获取验证数据
                data, info = st.session_state.predictor.get_stock_data(validate_stock)
                if data is None:
                    st.error(f"无法获取股票 {validate_stock} 的数据")
                else:
                    st.success(info)

                    # 进行预测
                    predictions, true_values = st.session_state.predictor.predict(data)

                    if predictions is not None:
                        # 评估模型
                        metrics = st.session_state.predictor.evaluate_model(true_values, predictions)

                        # 显示评估结果
                        st.subheader("验证结果")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("MAE", f"{metrics['MAE']:.4f}")
                        col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
                        col3.metric("R² Score", f"{metrics['R2']:.4f}")
                        col4.metric("方向准确率", f"{metrics['Direction_Accuracy']:.2f}%")

                        # 绘制验证结果
                        train_size = int(len(data) * 0.8)
                        test_dates = data.index[train_size + time_window:]

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=test_dates, y=true_values,
                            mode='lines', name='真实价格',
                            line=dict(color='blue', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=test_dates, y=predictions,
                            mode='lines', name='预测价格',
                            line=dict(color='red', width=1, dash='dash')
                        ))
                        fig.update_layout(
                            title=f'{validate_stock} 验证结果',
                            xaxis_title='日期',
                            yaxis_title='价格（元）',
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("请先训练模型或输入股票代码")

with tab3:
    st.header("未来预测")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("预测参数")
        predict_stock = st.text_input("预测股票代码", value="600519")
        predict_days = st.slider("预测天数", min_value=5, max_value=60, value=30)
        end_date = st.date_input("数据截止日期", value=datetime.now())

    with col2:
        st.subheader("预测说明")
        st.info("""
        基于历史数据预测未来股票价格走势。
        预测结果仅供参考，投资需谨慎。
        """)

    if st.button("开始预测", type="primary"):
        if predict_stock.strip() and st.session_state.model_trained:
            with st.spinner("正在进行预测..."):
                # 获取数据
                end_date_str = end_date.strftime("%Y%m%d")
                data, info = st.session_state.predictor.get_stock_data(predict_stock, end_date=end_date_str)

                if data is None:
                    st.error(f"无法获取股票 {predict_stock} 的数据")
                else:
                    st.success(info)

                    # 进行未来预测
                    future_predictions = st.session_state.predictor.predict_future(data, days=predict_days)

                    if future_predictions is not None:
                        # 创建未来日期
                        last_date = data.index[-1]
                        future_dates = [last_date + timedelta(days=i) for i in range(1, predict_days + 1)]

                        # 绘制预测结果
                        fig = make_subplots(rows=2, cols=1, subplot_titles=('完整视图', '预测详情'))

                        # 历史数据（最近180天）
                        recent_data = data.tail(180)
                        fig.add_trace(go.Scatter(
                            x=recent_data.index, y=recent_data['收盘'],
                            mode='lines', name='历史价格',
                            line=dict(color='blue', width=2)
                        ), row=1, col=1)

                        # 未来预测
                        fig.add_trace(go.Scatter(
                            x=future_dates, y=future_predictions,
                            mode='lines+markers', name='未来预测',
                            line=dict(color='red', width=2, dash='dash')
                        ), row=1, col=1)

                        fig.add_vline(x=last_date, line_dash="dash", line_color="gray", row=1, col=1)

                        # 预测详情
                        fig.add_trace(go.Scatter(
                            x=future_dates, y=future_predictions,
                            mode='lines+markers+text', name='预测价格',
                            line=dict(color='red', width=3),
                            text=[f'{price:.2f}' for price in future_predictions],
                            textposition="top center"
                        ), row=2, col=1)

                        fig.update_layout(
                            title=f'{predict_stock} 未来{predict_days}天价格预测',
                            height=600,
                            showlegend=True
                        )

                        fig.update_xaxes(title_text="日期", row=2, col=1)
                        fig.update_yaxes(title_text="价格（元）", row=1, col=1)
                        fig.update_yaxes(title_text="价格（元）", row=2, col=1)

                        st.plotly_chart(fig, use_container_width=True)

                        # 显示预测数据表
                        st.subheader("预测数据")
                        prediction_df = pd.DataFrame({
                            '日期': future_dates,
                            '预测价格': future_predictions
                        })
                        st.dataframe(prediction_df.style.format({'预测价格': '{:.2f}'}))
        else:
            st.error("请先训练模型或输入股票代码")

with tab4:
    st.header("量化策略回测")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("回测参数")
        backtest_stock = st.text_input("回测股票代码", value="000001")

        # 策略选择
        strategy_type = st.selectbox(
            "选择回测策略",
            ["simple", "rsi_based", "ma_crossover", "combined", "momentum"],
            format_func=lambda x: {
                "simple": "简单预测策略",
                "rsi_based": "RSI策略",
                "ma_crossover": "移动平均线策略",
                "combined": "综合策略",
                "momentum": "动量策略"
            }[x]
        )

        initial_capital = st.number_input("初始资金（元）", min_value=10000, max_value=1000000, value=100000, step=10000)
        transaction_cost = st.slider("交易成本（%）", min_value=0.0, max_value=0.5, value=0.1, step=0.05) / 100

        # 策略特定参数
        if strategy_type in ["rsi_based", "combined"]:
            col1a, col1b = st.columns(2)
            with col1a:
                rsi_oversold = st.slider("RSI超卖线", min_value=10, max_value=40, value=30)
            with col1b:
                rsi_overbought = st.slider("RSI超买线", min_value=60, max_value=90, value=70)

        if strategy_type in ["ma_crossover", "combined"]:
            col2a, col2b = st.columns(2)
            with col2a:
                ma_short = st.slider("短期均线", min_value=3, max_value=20, value=5)
            with col2b:
                ma_long = st.slider("长期均线", min_value=10, max_value=50, value=20)

        # 风险控制参数
        col3a, col3b = st.columns(2)
        with col3a:
            stop_loss = st.slider("止损比例（%）", min_value=1.0, max_value=20.0, value=5.0, step=0.5) / 100
        with col3b:
            take_profit = st.slider("止盈比例（%）", min_value=5.0, max_value=50.0, value=10.0, step=1.0) / 100

    with col2:
        st.subheader("策略说明")
        strategy_descriptions = {
            "simple": "基于预测价格方向进行买卖的简单策略",
            "rsi_based": "结合RSI超买超卖和预测信号的策略",
            "ma_crossover": "基于移动平均线交叉和预测信号的策略",
            "combined": "综合多种技术指标和预测信号的策略",
            "momentum": "基于价格动量和预测信号的策略"
        }
        st.info(strategy_descriptions.get(strategy_type, ""))

        st.subheader("风险提示")
        st.warning("""
        - 回测结果基于历史数据，不代表未来表现
        - 实际交易中应考虑更多因素
        - 投资有风险，入市需谨慎
        """)

    if st.button("开始回测", type="primary"):
        if backtest_stock.strip() and st.session_state.model_trained:
            with st.spinner("正在进行回测..."):
                # 获取数据
                data, info = st.session_state.predictor.get_stock_data(backtest_stock)
                if data is None:
                    st.error(f"无法获取股票 {backtest_stock} 的数据")
                else:
                    st.success(info)

                    # 进行预测
                    predictions, true_values = st.session_state.predictor.predict(data)

                    if predictions is not None:
                        # 执行回测
                        backtest_result, message = st.session_state.predictor.backtest_strategy(
                            data, predictions, strategy_type, initial_capital, transaction_cost,
                            rsi_oversold, rsi_overbought, ma_short, ma_long, stop_loss, take_profit
                        )

                        if backtest_result:
                            st.success(message)

                            # 显示回测结果
                            st.subheader("回测结果")

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("策略总收益", f"{backtest_result['total_return']:.2f}%")
                            col2.metric("年化收益", f"{backtest_result['annual_return']:.2f}%")
                            col3.metric("买入持有收益", f"{backtest_result['buy_hold_return']:.2f}%")
                            col4.metric("胜率", f"{backtest_result['win_rate']:.2f}%")

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("最大回撤", f"{backtest_result['max_drawdown']:.2f}%")
                            col2.metric("夏普比率", f"{backtest_result['sharpe_ratio']:.2f}")
                            col3.metric("交易次数", f"{backtest_result['total_trades']}")
                            strategy_name = {
                                "simple": "简单预测",
                                "rsi_based": "RSI策略",
                                "ma_crossover": "均线策略",
                                "combined": "综合策略",
                                "momentum": "动量策略"
                            }.get(backtest_result['strategy_type'], backtest_result['strategy_type'])
                            col4.metric("策略类型", strategy_name)

                            # 绘制回测结果
                            if len(backtest_result['dates']) > 0 and len(backtest_result['portfolio_values']) > 0:
                                fig = go.Figure()

                                # 策略净值曲线
                                fig.add_trace(go.Scatter(
                                    x=backtest_result['dates'],
                                    y=backtest_result['portfolio_values'],
                                    mode='lines',
                                    name='策略净值',
                                    line=dict(color='blue', width=2)
                                ))

                                # 买入持有净值曲线
                                buy_hold_values = [initial_capital * (
                                            1 + backtest_result['buy_hold_return'] / 100 * i / len(
                                        backtest_result['dates']))
                                                   for i in range(len(backtest_result['dates']))]
                                fig.add_trace(go.Scatter(
                                    x=backtest_result['dates'],
                                    y=buy_hold_values,
                                    mode='lines',
                                    name='买入持有',
                                    line=dict(color='green', width=2, dash='dash')
                                ))

                                # 标记交易点
                                buy_dates = []
                                buy_prices = []
                                sell_dates = []
                                sell_prices = []

                                for trade in backtest_result['trades']:
                                    if trade[0] == 'BUY':
                                        buy_dates.append(trade[1])
                                        buy_prices.append(trade[2])
                                    elif trade[0] == 'SELL':
                                        sell_dates.append(trade[1])
                                        sell_prices.append(trade[2])

                                if buy_dates:
                                    fig.add_trace(go.Scatter(
                                        x=buy_dates,
                                        y=buy_prices,
                                        mode='markers',
                                        name='买入点',
                                        marker=dict(color='green', size=10, symbol='triangle-up')
                                    ))

                                if sell_dates:
                                    fig.add_trace(go.Scatter(
                                        x=sell_dates,
                                        y=sell_prices,
                                        mode='markers',
                                        name='卖出点',
                                        marker=dict(color='red', size=10, symbol='triangle-down')
                                    ))

                                fig.update_layout(
                                    title=f'{backtest_stock} {strategy_name}回测结果',
                                    xaxis_title='日期',
                                    yaxis_title='投资组合价值（元）',
                                    hovermode='x unified',
                                    height=500
                                )

                                st.plotly_chart(fig, use_container_width=True)

                                # 显示技术指标图表
                                st.subheader("技术指标")
                                tech_fig = make_subplots(rows=2, cols=1, subplot_titles=('价格与均线', 'RSI指标'))

                                # 价格和均线
                                tech_fig.add_trace(go.Scatter(
                                    x=data.index,
                                    y=data['收盘'],
                                    mode='lines',
                                    name='收盘价',
                                    line=dict(color='black', width=1)
                                ), row=1, col=1)

                                if 'MA5' in data.columns:
                                    tech_fig.add_trace(go.Scatter(
                                        x=data.index,
                                        y=data['MA5'],
                                        mode='lines',
                                        name='MA5',
                                        line=dict(color='blue', width=1)
                                    ), row=1, col=1)

                                if 'MA20' in data.columns:
                                    tech_fig.add_trace(go.Scatter(
                                        x=data.index,
                                        y=data['MA20'],
                                        mode='lines',
                                        name='MA20',
                                        line=dict(color='red', width=1)
                                    ), row=1, col=1)

                                # RSI
                                if 'RSI' in data.columns:
                                    tech_fig.add_trace(go.Scatter(
                                        x=data.index,
                                        y=data['RSI'],
                                        mode='lines',
                                        name='RSI',
                                        line=dict(color='purple', width=1)
                                    ), row=2, col=1)

                                    # 添加超买超卖线
                                    tech_fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                                    tech_fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

                                tech_fig.update_layout(height=600, showlegend=True)
                                tech_fig.update_yaxes(title_text="价格", row=1, col=1)
                                tech_fig.update_yaxes(title_text="RSI", row=2, col=1)

                                st.plotly_chart(tech_fig, use_container_width=True)
                            else:
                                st.warning("回测数据不足，无法绘制图表")

                            # 显示交易记录
                            if backtest_result['trades']:
                                st.subheader("交易记录")
                                trades_df = pd.DataFrame(backtest_result['trades'],
                                                         columns=['操作', '日期', '价格', '备注'])
                                st.dataframe(trades_df.style.format({'价格': '{:.2f}'}))
                        else:
                            st.error(message)
        else:
            st.error("请先训练模型或输入股票代码")

with tab5:
    st.header("模型管理")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("保存模型")
        save_path = st.text_input("模型保存路径", value="saved_models/stock_predictor",
                                  help="请输入文件夹路径，例如: saved_models/my_model")

        st.info("""
        **保存说明:**
        - 模型将保存为一个文件夹和一个配置文件
        - 例如: 输入 `saved_models/my_model` 将创建:
          - `saved_models/my_model/` (模型文件夹)
          - `saved_models/my_model_config.pkl` (配置文件)
        """)

        if st.button("保存模型", type="primary"):
            if st.session_state.model_trained:
                success, message = st.session_state.predictor.save_model(save_path)
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.error("没有训练好的模型可保存")

    with col2:
        st.subheader("加载模型")
        load_path = st.text_input("模型加载路径", value="saved_models/stock_predictor",
                                  help="请输入模型文件夹路径，例如: saved_models/my_model")

        st.info("""
        **加载说明:**
        - 请输入模型文件夹的路径
        - 例如: 输入 `saved_models/my_model` 将加载:
          - `saved_models/my_model/` (模型文件夹)
          - `saved_models/my_model_config.pkl` (配置文件)
        - 请确保这两个文件都存在
        """)

        # 显示当前目录下的模型文件
        if st.button("查看可用模型"):
            model_dirs = []
            if os.path.exists("saved_models"):
                for item in os.listdir("saved_models"):
                    item_path = os.path.join("saved_models", item)
                    if os.path.isdir(item_path):
                        config_file = os.path.join("saved_models", f"{item}_config.pkl")
                        if os.path.exists(config_file):
                            model_dirs.append(item)

            if model_dirs:
                st.write("可用的模型:")
                for model_dir in model_dirs:
                    st.write(f"- saved_models/{model_dir}")
            else:
                st.write("没有找到可用的模型")

        if st.button("加载模型", type="primary"):
            success, message = st.session_state.predictor.load_model(load_path)
            if success:
                st.session_state.model_trained = True
                st.success(message)
            else:
                st.error(message)

    # 模型信息
    st.subheader("模型信息")
    if st.session_state.model_trained:
        st.success("✅ 模型已加载")
        st.write(f"- 模型类型: {st.session_state.predictor.model_type}")
        st.write(f"- 时间窗口: {st.session_state.predictor.time_window}天")
        st.write(f"- 输入特征: {', '.join(st.session_state.predictor.features)}")
        if st.session_state.training_history:
            st.write(f"- 最终训练损失: {st.session_state.training_history.history['loss'][-1]:.4f}")
            st.write(f"- 最终验证损失: {st.session_state.training_history.history['val_loss'][-1]:.4f}")
    else:
        st.warning("⏳ 模型未加载")

    # 数据缓存管理
    st.subheader("数据缓存管理")
    if st.button("清除数据缓存"):
        cache_dir = "stock_data_cache"
        if os.path.exists(cache_dir):
            for file in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    st.error(f"删除文件 {file_path} 时出错: {e}")
            st.success("数据缓存已清除")
        else:
            st.info("数据缓存目录不存在")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>📊 智能股票预测系统 | 基于深度学习的时间序列预测与量化回测</p>
        <p><small>注意：本系统预测结果仅供参考，不构成投资建议。股市有风险，投资需谨慎。</small></p>
    </div>
    """,
    unsafe_allow_html=True
)
