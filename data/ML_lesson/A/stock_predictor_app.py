import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, MultiHeadAttention, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import akshare as ak
import warnings
import pickle
import os
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')

# 设置页面配置
st.set_page_config(
    page_title="智能股票预测系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


class StockPredictor:
    def __init__(self, time_window=30):
        self.time_window = time_window
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.features = ["开盘", "最高", "最低", "收盘", "成交量"]
        self.close_idx = self.features.index("收盘")
        self.is_trained = False

    def get_stock_data(self, stock_code, start_date="20200101", end_date=None):
        """获取股票数据"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y%m%d")

        try:
            df = ak.stock_zh_a_hist(
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )

            if df is not None and len(df) > 0:
                # 数据预处理
                df["日期"] = pd.to_datetime(df["日期"])
                df = df.sort_values("日期")
                df.set_index("日期", inplace=True)

                # 选择特征
                available_features = [col for col in self.features if col in df.columns]
                data = df[available_features].dropna()

                return data, f"成功获取股票 {stock_code} 数据，共 {len(data)} 行"
            else:
                return None, f"股票 {stock_code} 返回空数据"

        except Exception as e:
            return None, f"获取股票 {stock_code} 时出错: {e}"

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

        return df, f"创建 {stock_name} 模拟数据，共 {len(df)} 行"

    def prepare_data(self, data):
        """准备训练数据"""
        scaled_data = self.scaler.fit_transform(data)

        def create_sequences(data, time_window, target_idx):
            X, y = [], []
            for i in range(time_window, len(data)):
                X.append(data[i - time_window:i])
                y.append(data[i, target_idx])
            return np.array(X), np.array(y)

        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        train_X, train_y = create_sequences(train_data, self.time_window, self.close_idx)
        test_X, test_y = create_sequences(test_data, self.time_window, self.close_idx)

        return train_X, train_y, test_X, test_y, train_size

    def build_model(self, input_shape):
        """构建LSTM+Attention模型"""
        inputs = Input(shape=input_shape)
        lstm_out = LSTM(64, return_sequences=True, activation='tanh')(inputs)
        lstm_out = Dropout(0.2)(lstm_out)

        attention_out = MultiHeadAttention(num_heads=4, key_dim=32)(lstm_out, lstm_out)
        combined = lstm_out + attention_out

        last_step = combined[:, -1, :]
        output = Dense(32, activation='relu')(last_step)
        output = Dropout(0.2)(output)
        output = Dense(1, activation='linear')(output)

        model = Model(inputs=inputs, outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        return model

    def train_model(self, stock_codes, epochs=50, use_simulated=False):
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

            train_X, train_y, test_X, test_y, _ = self.prepare_data(data)

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
        self.model = self.build_model((self.time_window, len(self.features)))

        # 训练模型
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(
            train_X, train_y,
            epochs=epochs,
            batch_size=32,
            validation_data=(test_X, test_y),
            verbose=0,
            callbacks=[early_stopping]
        )

        self.is_trained = True

        return history, train_X, train_y, test_X, test_y, stock_data_info

    def predict(self, data):
        """对给定数据进行预测"""
        if not self.is_trained:
            return None, "模型未训练"

        scaled_data = self.scaler.transform(data)

        X, y = [], []
        for i in range(self.time_window, len(scaled_data)):
            X.append(scaled_data[i - self.time_window:i])
            y.append(scaled_data[i, self.close_idx])

        X, y = np.array(X), np.array(y)

        predictions = self.model.predict(X, verbose=0)

        # 反归一化
        predictions_inv = self.inverse_transform_pred(predictions)
        y_inv = self.inverse_transform_pred(y.reshape(-1, 1))

        return predictions_inv, y_inv

    def predict_future(self, data, days=30):
        """预测未来价格走势"""
        if not self.is_trained:
            return None, "模型未训练"

        last_sequence = data[-self.time_window:]
        last_sequence_scaled = self.scaler.transform(last_sequence)

        future_predictions = []
        current_sequence = last_sequence_scaled.copy()

        for _ in range(days):
            next_pred = self.model.predict(current_sequence.reshape(1, self.time_window, len(self.features)), verbose=0)
            future_predictions.append(next_pred[0, 0])

            new_day = current_sequence[-1].copy()
            new_day[self.close_idx] = next_pred[0, 0]
            current_sequence = np.vstack([current_sequence[1:], new_day])

        future_predictions = self.inverse_transform_pred(np.array(future_predictions).reshape(-1, 1))

        return future_predictions

    def inverse_transform_pred(self, y_pred):
        """反归一化预测结果"""
        y_reshaped = np.zeros(shape=(len(y_pred), len(self.features)))
        y_reshaped[:, self.close_idx] = y_pred.flatten()
        return self.scaler.inverse_transform(y_reshaped)[:, self.close_idx]

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

    def save_model(self, filepath):
        """保存模型和配置"""
        if self.model is None:
            return False, "没有训练好的模型可保存"

        # 创建目录
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        # 保存模型
        model_path = filepath if filepath.endswith('.h5') else filepath + '.h5'
        self.model.save(model_path)

        # 保存scaler和其他配置
        config = {
            'time_window': self.time_window,
            'features': self.features,
            'close_idx': self.close_idx,
            'is_trained': self.is_trained,
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

        return True, f"模型已保存到 {model_path}"

    def load_model(self, filepath):
        """加载模型和配置"""
        model_path = filepath if filepath.endswith('.h5') else filepath + '.h5'
        config_path = filepath.replace('.h5', '_config.pkl') if filepath.endswith('.h5') else filepath + '_config.pkl'

        if not os.path.exists(model_path) or not os.path.exists(config_path):
            return False, "模型文件不存在"

        try:
            # 加载模型
            self.model = tf.keras.models.load_model(model_path)

            # 加载配置
            with open(config_path, 'rb') as f:
                config = pickle.load(f)

            self.time_window = config['time_window']
            self.features = config['features']
            self.close_idx = config['close_idx']
            self.is_trained = config['is_trained']

            # 恢复scaler
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.scaler.min_ = config['scaler_params']['min_']
            self.scaler.scale_ = config['scaler_params']['scale_']
            self.scaler.data_min_ = config['scaler_params']['data_min_']
            self.scaler.data_max_ = config['scaler_params']['data_max_']
            self.scaler.data_range_ = config['scaler_params']['data_range_']

            return True, f"模型已从 {model_path} 加载"

        except Exception as e:
            return False, f"加载模型时出错: {e}"


# 初始化预测器
if 'predictor' not in st.session_state:
    st.session_state.predictor = StockPredictor()
    st.session_state.model_trained = False
    st.session_state.training_history = None

# 界面设计
st.title("📈 智能股票预测系统")
st.markdown("使用LSTM+Attention神经网络进行多股票训练和预测")

# 侧边栏
st.sidebar.header("模型配置")

# 时间窗口设置
time_window = st.sidebar.slider("时间窗口大小", min_value=10, max_value=60, value=30, step=5)
st.session_state.predictor.time_window = time_window

# 主界面选项卡
tab1, tab2, tab3, tab4 = st.tabs(["🚀 模型训练", "🔍 股票验证", "🔮 未来预测", "💾 模型管理"])

with tab1:
    st.header("模型训练")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("训练参数")
        train_stocks = st.text_area(
            "训练股票代码（用逗号分隔）",
            value="600519,000858,300750",
            help="例如：600519,000858,300750"
        )
        epochs = st.number_input("训练轮数 (Epochs)", min_value=10, max_value=500, value=50)
        use_simulated = st.checkbox("使用模拟数据（当真实数据不可用时）", value=True)

    with col2:
        st.subheader("训练状态")
        if st.session_state.model_trained:
            st.success("✅ 模型已训练完成")
            if st.session_state.training_history:
                st.write(f"最终训练损失: {st.session_state.training_history.history['loss'][-1]:.4f}")
                st.write(f"最终验证损失: {st.session_state.training_history.history['val_loss'][-1]:.4f}")
        else:
            st.warning("⏳ 模型未训练")

    if st.button("开始训练模型", type="primary"):
        if train_stocks.strip():
            stock_list = [s.strip() for s in train_stocks.split(',') if s.strip()]

            with st.spinner("正在训练模型，请稍候..."):
                progress_bar = st.progress(0)

                history, train_X, train_y, test_X, test_y, stock_info = st.session_state.predictor.train_model(
                    stock_list, epochs=epochs, use_simulated=use_simulated
                )

                st.session_state.training_history = history
                st.session_state.model_trained = True
                progress_bar.progress(100)

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
    st.header("模型管理")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("保存模型")
        save_path = st.text_input("模型保存路径", value="saved_models/stock_predictor")

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
        load_path = st.text_input("模型加载路径", value="saved_models/stock_predictor.h5")

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
        st.write(f"- 时间窗口: {st.session_state.predictor.time_window}天")
        st.write(f"- 输入特征: {', '.join(st.session_state.predictor.features)}")
        st.write(f"- 模型结构: LSTM + MultiHeadAttention")
    else:
        st.warning("⏳ 模型未加载")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>📊 智能股票预测系统 | 基于深度学习的时间序列预测</p>
        <p><small>注意：本系统预测结果仅供参考，不构成投资建议。股市有风险，投资需谨慎。</small></p>
    </div>
    """,
    unsafe_allow_html=True
)