# 导入核心库
import streamlit as st
import akshare as ak
import pandas as pd
import sqlite3
import random
import time
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import numpy as
# ===================== 机器学习库 =====================
try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR
    from xgboost import XGBRegressor

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("机器学习库未安装，请运行：pip install scikit-learn xgboost")

# TensorFlow/Keras LSTM模型
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Attention, Concatenate
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("TensorFlow未安装，请运行：pip install tensorflow")


# 处理fake_useragent的替代方案
class RandomUserAgent:
    """自定义UserAgent生成器"""

    def __init__(self):
        self.browsers = [
            # Chrome
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
            # Firefox
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:120.0) Gecko/20100101 Firefox/120.0",
            # Safari
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        ]

    def random(self):
        return random.choice(self.browsers)


# ===================== 爬虫伪装配置 =====================
def setup_session_with_retry():
    """配置带重试和伪装头的requests session"""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()

    # 配置重试策略
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # 使用自定义UserAgent
    ua = RandomUserAgent()
    headers = {
        'User-Agent': ua.random(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }
    session.headers.update(headers)

    return session


# ===================== 数据库初始化 =====================
def init_db(db_path="quant_analysis.db"):
    """初始化SQLite数据库，创建股票数据表（复合主键避免重复）"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 数据表结构：股票代码、日期、开高低收、成交量、成交额
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS stock_data (
        stock_code TEXT,
        date DATE,
        open FLOAT,
        close FLOAT,
        high FLOAT,
        low FLOAT,
        volume FLOAT,
        amount FLOAT,
        freq TEXT DEFAULT 'daily',
        update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (stock_code, date, freq)
    )
    ''')

    # 创建机器学习预测结果表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ml_predictions (
        prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        stock_code TEXT,
        model_name TEXT,
        prediction_date DATE,
        prediction_days INTEGER,
        mse FLOAT,
        mae FLOAT,
        r2 FLOAT,
        predictions TEXT,  -- JSON格式存储预测值
        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 创建索引提高查询速度
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_data(stock_code, date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON stock_data(date)')

    conn.commit()
    return conn


# ===================== 数据抓取 =====================
def get_stock_data(stock_code, start_date, end_date, freq="daily"):
    """
    抓取A股股票数据（支持日线/周线/月线，前复权）
    """
    try:
        # 随机延时（1-3秒）
        time.sleep(random.uniform(1, 3))

        # 自动补全市场标识
        if stock_code.startswith(('6', '9', '5')):
            stock_code_full = f"{stock_code}.SH"
        elif stock_code.startswith(('0', '3', '2')):
            stock_code_full = f"{stock_code}.SZ"
        elif stock_code.startswith('8'):
            stock_code_full = f"{stock_code}.BJ"
        else:
            st.error("暂不支持该市场代码（仅支持沪深A股）")
            return None

        # 转换频率参数
        period_map = {
            "daily": "daily",
            "weekly": "weekly",
            "monthly": "monthly"
        }

        if freq not in period_map:
            freq = "daily"

        # 调用akshare获取数据
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period=period_map[freq],
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )

        if df is None or df.empty:
            st.warning(f"⚠️ 未获取到{stock_code}在{start_date}到{end_date}的数据")
            return None

        # 数据标准化处理
        df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"
        }, inplace=True)

        df["stock_code"] = stock_code
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["freq"] = freq

        # 转换数据类型
        numeric_cols = ["open", "close", "high", "low", "volume", "amount"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 添加技术指标
        df = add_technical_indicators(df)

        df.reset_index(drop=True, inplace=True)

        st.success(f"✅ 成功获取{stock_code} {freq}数据，共{len(df)}条记录")
        return df

    except Exception as e:
        st.error(f"❌ 抓取失败：{str(e)}")
        import traceback
        st.error(f"详细错误：{traceback.format_exc()}")
        return None


def add_technical_indicators(df):
    """添加技术指标"""
    df = df.copy()

    # 移动平均线
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()

    # 相对强弱指数 (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 布林带
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std

    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # 成交量指标
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # 价格变化率
    df['price_change'] = df['close'].pct_change() * 100
    df['price_change_5d'] = df['close'].pct_change(5) * 100

    # 填充NaN值
    df = df.fillna(method='bfill').fillna(method='ffill')

    return df


# ===================== 数据显示 =====================
def show_stock_data(df, stock_code):
    """可视化展示股票数据"""
    if df is None or df.empty:
        st.warning("📭 暂无数据可显示")
        return

    # 1. 数据表格
    st.subheader(f"{stock_code} 原始数据")
    st.dataframe(df, use_container_width=True)

    # 下载按钮
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 下载CSV",
        data=csv,
        file_name=f"{stock_code}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    # 2. 基本统计
    st.subheader("📊 数据统计摘要")
    stats = df[["open", "close", "high", "low", "volume", "amount"]].describe()
    st.write(stats)

    # 3. 收盘价走势
    st.subheader("📈 收盘价走势")
    fig_price = px.line(
        df, x="date", y="close",
        title=f"{stock_code} 收盘价走势（前复权）",
        labels={"date": "日期", "close": "收盘价（元）"},
        template="plotly_white"
    )

    # 添加移动平均线
    if 'MA20' in df.columns:
        fig_price.add_scatter(x=df['date'], y=df['MA20'],
                              mode='lines', name='20日均线',
                              line=dict(color='orange', width=2))

    if 'MA60' in df.columns:
        fig_price.add_scatter(x=df['date'], y=df['MA60'],
                              mode='lines', name='60日均线',
                              line=dict(color='green', width=2))

    st.plotly_chart(fig_price, use_container_width=True)

    # 4. K线图
    st.subheader("📉 K线图")
    fig_kline = go.Figure(data=[go.Candlestick(
        x=df["date"],
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="K线",
        increasing_line_color="#ff4b4b",
        decreasing_line_color="#009966"
    )])

    # 添加成交量
    fig_kline.add_trace(go.Bar(
        x=df["date"],
        y=df["volume"],
        name="成交量",
        marker_color="rgba(100, 100, 200, 0.5)",
        yaxis="y2"
    ))

    fig_kline.update_layout(
        title=f"{stock_code} K线图（前复权）",
        xaxis_title="日期",
        yaxis_title="价格（元）",
        yaxis2=dict(
            title="成交量",
            overlaying="y",
            side="right",
            showgrid=False
        ),
        template="plotly_white",
        height=600,
        hovermode="x unified"
    )
    st.plotly_chart(fig_kline, use_container_width=True)

    # 5. 技术指标图
    st.subheader("📊 技术指标")

    # RSI图
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df['date'], y=df['RSI'], name='RSI', line=dict(color='purple')))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线")
    fig_rsi.update_layout(title="RSI指标", height=300)
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD图
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df['date'], y=df['MACD'], name='MACD', line=dict(color='blue')))
    fig_macd.add_trace(go.Scatter(x=df['date'], y=df['MACD_signal'], name='信号线', line=dict(color='orange')))

    # MACD柱状图
    colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
    fig_macd.add_trace(go.Bar(x=df['date'], y=df['MACD_hist'], name='MACD柱', marker_color=colors))

    fig_macd.update_layout(title="MACD指标", height=300)
    st.plotly_chart(fig_macd, use_container_width=True)


# ===================== 机器学习预测模块 =====================
class StockPredictor:
    """股票价格预测器"""

    def __init__(self):
        self.scaler = MinMaxScaler()

    def prepare_data(self, df, target_col='close', feature_cols=None,
                     lookback=60, forecast_days=30):
        """准备时间序列数据"""
        if feature_cols is None:
            feature_cols = ['close', 'volume', 'MA5', 'MA20', 'RSI', 'MACD']

        # 选择特征列
        available_cols = [col for col in feature_cols if col in df.columns]
        if len(available_cols) == 0:
            available_cols = ['close']

        data = df[available_cols].values

        # 数据标准化
        data_scaled = self.scaler.fit_transform(data)

        # 创建时间序列样本
        X, y = [], []
        for i in range(lookback, len(data_scaled) - forecast_days):
            X.append(data_scaled[i - lookback:i])
            y.append(data_scaled[i:i + forecast_days, 0])  # 预测close

        return np.array(X), np.array(y)

    def split_data(self, X, y, test_size=0.2):
        """分割数据集"""
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test

    def build_lstm_model(self, input_shape, lstm_units=50, dropout_rate=0.2):
        """构建标准LSTM模型"""
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(input_shape[1])  # 预测多个时间步
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mse', metrics=['mae'])
        return model

    def build_lstm_attention_model(self, input_shape, lstm_units=64, dropout_rate=0.3):
        """构建LSTM with Attention模型"""
        # 编码器
        inputs = Input(shape=input_shape)
        lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
        lstm_out = Dropout(dropout_rate)(lstm_out)

        # Attention机制
        attention = Attention()([lstm_out, lstm_out])
        attention = LSTM(lstm_units)(attention)
        attention = Dropout(dropout_rate)(attention)

        # 解码器
        dense1 = Dense(128, activation='relu')(attention)
        dense1 = Dropout(dropout_rate)(dense1)
        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = Dropout(dropout_rate)(dense2)

        # 输出层
        outputs = Dense(input_shape[1])(dense2)  # 预测多个时间步

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mse', metrics=['mae'])
        return model

    def build_bilstm_model(self, input_shape, lstm_units=64, dropout_rate=0.3):
        """构建双向LSTM模型"""
        model = Sequential([
            Bidirectional(LSTM(lstm_units, return_sequences=True), input_shape=input_shape),
            Dropout(dropout_rate),
            Bidirectional(LSTM(lstm_units)),
            Dropout(dropout_rate),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            Dense(64, activation='relu'),
            Dense(input_shape[1])
        ])

        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mse', metrics=['mae'])
        return model

    def train_traditional_model(self, model_name, X_train, y_train, **params):
        """训练传统机器学习模型"""
        X_train_flat = X_train.reshape(X_train.shape[0], -1)

        if model_name == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 10),
                random_state=42
            )
        elif model_name == 'svr':
            model = SVR(
                kernel=params.get('kernel', 'rbf'),
                C=params.get('C', 1.0),
                epsilon=params.get('epsilon', 0.1)
            )
        elif model_name == 'xgboost':
            model = XGBRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=42
            )
        else:
            raise ValueError(f"不支持的传统模型: {model_name}")

        model.fit(X_train_flat, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test, model_type='lstm'):
        """评估模型性能"""
        if model_type == 'lstm':
            y_pred = model.predict(X_test)
        else:
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_pred = model.predict(X_test_flat)

        # 计算指标
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return y_pred, {'mse': mse, 'mae': mae, 'r2': r2}

    def predict_future(self, model, last_sequence, forecast_days, model_type='lstm'):
        """预测未来股价"""
        predictions = []
        current_sequence = last_sequence.copy()

        for _ in range(forecast_days):
            if model_type == 'lstm':
                pred = model.predict(current_sequence.reshape(1, *current_sequence.shape))[0]
            else:
                pred = model.predict(current_sequence.flatten().reshape(1, -1))[0]

            predictions.append(pred[0])  # 只取close的预测

            # 更新序列
            new_row = current_sequence[-1].copy()
            new_row[0] = pred[0]  # 更新close值
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = new_row

        return np.array(predictions)


# ===================== 数据存储 =====================
def save_stock_data(df, stock_code, save_type, db_conn=None):
    """存储数据"""
    if df is None or df.empty:
        st.warning("📭 暂无数据可存储")
        return False

    try:
        import os
        os.makedirs("data", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stock_code}_stock_{timestamp}"

        if save_type == "csv":
            filepath = f"data/{filename}.csv"
            df.to_csv(filepath, index=False, encoding="utf-8")
            st.success(f"💾 已保存为CSV：{filepath}")
            return True

        elif save_type == "excel":
            filepath = f"data/{filename}.xlsx"
            df.to_excel(filepath, index=False, engine="openpyxl")
            st.success(f"💾 已保存为Excel：{filepath}")
            return True

        elif save_type == "sqlite":
            if not db_conn:
                st.error("❌ SQLite连接未初始化")
                return False

            cursor = db_conn.cursor()
            freq = df.iloc[0]['freq'] if 'freq' in df.columns else 'daily'

            cursor.execute(
                "DELETE FROM stock_data WHERE stock_code = ? AND freq = ?",
                (stock_code, freq)
            )
            db_conn.commit()

            df.to_sql("stock_data", db_conn, if_exists="append", index=False)
            st.success(f"💾 已存入SQLite数据库，共{len(df)}条记录")
            return True

        else:
            st.error("❌ 存储类型错误")
            return False

    except Exception as e:
        st.error(f"❌ 存储失败：{str(e)}")
        return False


# ===================== 主界面 =====================
def main():
    # 页面基础配置
    st.set_page_config(
        page_title="量化分析系统 - 数据获取与预测",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 页面标题
    st.title("📈 量化分析系统 | 数据获取与股价预测")
    st.markdown("---")

    # 初始化数据库连接
    conn = init_db()

    # 侧边栏功能菜单
    st.sidebar.title("⚙️ 功能菜单")
    func_choice = st.sidebar.radio(
        "选择功能",
        ["数据抓取", "数据显示", "数据存储", "数据更新", "股价预测", "数据库管理"],
        index=0
    )

    # 会话状态初始化
    if "stock_data" not in st.session_state:
        st.session_state["stock_data"] = None
    if "stock_code" not in st.session_state:
        st.session_state["stock_code"] = "600000"
    if "freq" not in st.session_state:
        st.session_state["freq"] = "daily"
    if "predictor" not in st.session_state:
        st.session_state["predictor"] = StockPredictor()
    if "ml_model" not in st.session_state:
        st.session_state["ml_model"] = None

    # ========== 功能1：数据抓取 ==========
    if func_choice == "数据抓取":
        st.subheader("🔍 股票数据抓取")

        col1, col2 = st.columns([1, 2])
        with col1:
            stock_code = st.text_input(
                "股票代码",
                value="600000",
                help="如600000（浦发银行）、000001（平安银行）"
            )
            freq = st.selectbox(
                "数据频率",
                ["daily", "weekly", "monthly"],
                index=0
            )

        with col2:
            date_range = st.date_input(
                "选择日期范围",
                value=[datetime.now() - timedelta(days=365), datetime.now()],
                max_value=datetime.now()
            )

            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = datetime.now() - timedelta(days=365)
                end_date = datetime.now()

        if st.button("🚀 开始抓取", type="primary", use_container_width=True):
            with st.spinner(f"正在抓取{stock_code}的{freq}数据..."):
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")

                df = get_stock_data(stock_code, start_str, end_str, freq)

                if df is not None:
                    st.session_state["stock_data"] = df
                    st.session_state["stock_code"] = stock_code
                    st.session_state["freq"] = freq

                    st.subheader("📋 数据预览")
                    st.dataframe(df.head(), use_container_width=True)

    # ========== 功能2：数据显示 ==========
    elif func_choice == "数据显示":
        st.subheader("📊 数据可视化展示")

        if st.session_state["stock_data"] is None:
            st.warning("📭 请先抓取数据！")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📌 快速抓取示例（600000近1年）"):
                    start_str = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                    end_str = datetime.now().strftime("%Y-%m-%d")
                    df = get_stock_data("600000", start_str, end_str, "daily")
                    st.session_state["stock_data"] = df
                    st.session_state["stock_code"] = "600000"
                    st.rerun()
        else:
            show_stock_data(
                st.session_state["stock_data"],
                st.session_state["stock_code"]
            )

    # ========== 功能3：数据存储 ==========
    elif func_choice == "数据存储":
        st.subheader("💾 数据持久化存储")

        if st.session_state["stock_data"] is None:
            st.warning("📭 请先抓取数据！")
        else:
            save_type = st.radio(
                "存储类型",
                ["csv", "excel", "sqlite"],
                index=0,
                horizontal=True
            )

            if st.button("💾 开始存储", type="primary", use_container_width=True):
                with st.spinner("正在存储数据..."):
                    success = save_stock_data(
                        st.session_state["stock_data"],
                        st.session_state["stock_code"],
                        save_type,
                        conn
                    )

    # ========== 功能4：数据更新 ==========
    elif func_choice == "数据更新":
        st.subheader("🔄 增量更新数据")

        col1, col2 = st.columns(2)
        with col1:
            stock_code = st.text_input(
                "待更新股票代码",
                value=st.session_state["stock_code"]
            )

        with col2:
            freq = st.selectbox(
                "数据频率",
                ["daily", "weekly", "monthly"],
                index=0
            )

        if st.button("🔄 开始更新", type="primary", use_container_width=True):
            st.info("数据更新功能需要从原始代码中扩展实现")

    # ========== 功能5：股价预测 ==========
    elif func_choice == "股价预测":
        st.subheader("🤖 机器学习股价预测")

        if not ML_AVAILABLE:
            st.error("请先安装机器学习库：pip install scikit-learn xgboost")
            return

        if st.session_state["stock_data"] is None:
            st.warning("📭 请先抓取数据！")

            if st.button("📌 快速获取预测数据（000001）"):
                start_str = (datetime.now() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
                end_str = datetime.now().strftime("%Y-%m-%d")
                df = get_stock_data("000001", start_str, end_str, "daily")
                st.session_state["stock_data"] = df
                st.session_state["stock_code"] = "000001"
                st.rerun()
            return

        df = st.session_state["stock_data"]
        stock_code = st.session_state["stock_code"]

        # 预测参数设置
        st.subheader("⚙️ 预测参数设置")

        col1, col2, col3 = st.columns(3)
        with col1:
            model_type = st.selectbox(
                "模型类型",
                ["LSTM", "LSTM with Attention", "BiLSTM", "Random Forest", "XGBoost", "SVR"],
                index=0
            )

        with col2:
            lookback = st.slider("回看窗口", 10, 120, 60, help="使用多少天的历史数据预测")

        with col3:
            forecast_days = st.slider("预测天数", 5, 60, 30, help="预测未来多少天")

        # 模型特定参数
        st.subheader("🔧 模型参数")

        if model_type in ["LSTM", "LSTM with Attention", "BiLSTM"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                lstm_units = st.slider("LSTM单元数", 16, 256, 64)
            with col2:
                dropout_rate = st.slider("Dropout率", 0.0, 0.5, 0.2, 0.1)
            with col3:
                epochs = st.slider("训练轮数", 10, 200, 50)
        else:
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.slider("树的数量", 50, 500, 100)
            with col2:
                max_depth = st.slider("最大深度", 3, 20, 10)

        # 特征选择
        st.subheader("🎯 特征选择")
        available_features = [col for col in df.columns if col not in ['date', 'stock_code', 'freq']]
        selected_features = st.multiselect(
            "选择特征",
            available_features,
            default=['close', 'volume', 'MA20', 'RSI', 'MACD']
        )

        if st.button("🚀 开始训练与预测", type="primary", use_container_width=True):
            with st.spinner("正在准备数据..."):
                predictor = st.session_state["predictor"]

                # 准备数据
                X, y = predictor.prepare_data(
                    df,
                    feature_cols=selected_features,
                    lookback=lookback,
                    forecast_days=forecast_days
                )

                if len(X) == 0:
                    st.error("数据不足，请选择更长的历史数据")
                    return

                # 分割数据
                X_train, X_test, y_train, y_test = predictor.split_data(X, y, test_size=0.2)

                st.info(f"数据集大小: 训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本")

                # 训练模型
                with st.spinner("正在训练模型..."):
                    if model_type in ["LSTM", "LSTM with Attention", "BiLSTM"]:
                        # 深度学习模型
                        input_shape = (X_train.shape[1], X_train.shape[2])

                        if model_type == "LSTM":
                            model = predictor.build_lstm_model(input_shape, lstm_units, dropout_rate)
                        elif model_type == "LSTM with Attention":
                            model = predictor.build_lstm_attention_model(input_shape, lstm_units, dropout_rate)
                        else:  # BiLSTM
                            model = predictor.build_bilstm_model(input_shape, lstm_units, dropout_rate)

                        # 训练
                        history = model.fit(
                            X_train, y_train,
                            validation_split=0.2,
                            epochs=epochs,
                            batch_size=32,
                            verbose=0,
                            callbacks=[
                                EarlyStopping(patience=10, restore_best_weights=True),
                                ReduceLROnPlateau(factor=0.5, patience=5)
                            ]
                        )

                        # 显示训练历史
                        fig_history = go.Figure()
                        fig_history.add_trace(go.Scatter(y=history.history['loss'], name='训练损失'))
                        fig_history.add_trace(go.Scatter(y=history.history['val_loss'], name='验证损失'))
                        fig_history.update_layout(title="训练历史", xaxis_title="轮数", yaxis_title="损失")
                        st.plotly_chart(fig_history, use_container_width=True)

                    else:
                        # 传统机器学习模型
                        model_params = {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth
                        }

                        if model_type == "Random Forest":
                            model = predictor.train_traditional_model('random_forest', X_train, y_train, **model_params)
                        elif model_type == "XGBoost":
                            model = predictor.train_traditional_model('xgboost', X_train, y_train, **model_params)
                        elif model_type == "SVR":
                            model = predictor.train_traditional_model('svr', X_train, y_train, **model_params)

                # 评估模型
                with st.spinner("正在评估模型..."):
                    model_type_for_eval = 'lstm' if model_type in ["LSTM", "LSTM with Attention",
                                                                   "BiLSTM"] else 'traditional'
                    y_pred, metrics = predictor.evaluate_model(model, X_test, y_test, model_type_for_eval)

                    # 显示评估指标
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("均方误差 (MSE)", f"{metrics['mse']:.4f}")
                    with col2:
                        st.metric("平均绝对误差 (MAE)", f"{metrics['mae']:.4f}")
                    with col3:
                        st.metric("R² 分数", f"{metrics['r2']:.4f}")

                # 可视化预测结果
                st.subheader("📊 预测结果可视化")

                # 反标准化预测结果
                y_test_original = y_test * predictor.scaler.data_range_[0] + predictor.scaler.data_min_[0]
                y_pred_original = y_pred * predictor.scaler.data_range_[0] + predictor.scaler.data_min_[0]

                # 绘制预测对比图
                fig_pred = go.Figure()

                # 添加实际值
                for i in range(min(5, len(y_test))):
                    fig_pred.add_trace(go.Scatter(
                        y=y_test_original[i],
                        mode='lines',
                        name=f'实际值 {i + 1}',
                        line=dict(color='blue', width=2 if i == 0 else 1)
                    ))

                # 添加预测值
                for i in range(min(5, len(y_pred))):
                    fig_pred.add_trace(go.Scatter(
                        y=y_pred_original[i],
                        mode='lines+markers',
                        name=f'预测值 {i + 1}',
                        line=dict(color='red', dash='dash', width=2 if i == 0 else 1)
                    ))

                fig_pred.update_layout(
                    title="测试集预测对比",
                    xaxis_title="预测天数",
                    yaxis_title="股价",
                    height=400
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # 预测未来股价
                st.subheader("🔮 未来股价预测")

                # 使用最后的数据预测未来
                last_sequence = X[-1]
                future_predictions = predictor.predict_future(
                    model, last_sequence, forecast_days,
                    model_type_for_eval
                )

                # 反标准化
                future_predictions_original = future_predictions * predictor.scaler.data_range_[0] + \
                                              predictor.scaler.data_min_[0]

                # 生成未来日期
                last_date = df['date'].iloc[-1]
                future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]

                # 绘制未来预测图
                fig_future = go.Figure()

                # 历史数据
                fig_future.add_trace(go.Scatter(
                    x=df['date'].iloc[-100:],  # 显示最近100天
                    y=df['close'].iloc[-100:],
                    mode='lines',
                    name='历史收盘价',
                    line=dict(color='blue', width=2)
                ))

                # 未来预测
                fig_future.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_predictions_original,
                    mode='lines+markers',
                    name='未来预测',
                    line=dict(color='red', width=2)
                ))

                # 置信区间（简单版本）
                std_dev = np.std(future_predictions_original)
                fig_future.add_trace(go.Scatter(
                    x=future_dates + future_dates[::-1],
                    y=list(future_predictions_original + std_dev) + list(future_predictions_original - std_dev)[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='置信区间'
                ))

                fig_future.update_layout(
                    title=f"{stock_code} 未来{forecast_days}天股价预测",
                    xaxis_title="日期",
                    yaxis_title="股价（元）",
                    height=500
                )
                st.plotly_chart(fig_future, use_container_width=True)

                # 显示预测结果表格
                st.subheader("📋 预测结果详情")
                future_df = pd.DataFrame({
                    '日期': future_dates,
                    '预测股价': future_predictions_original,
                    '日变化': np.concatenate([[0], np.diff(future_predictions_original)]),
                    '累计变化': future_predictions_original - df['close'].iloc[-1]
                })
                st.dataframe(future_df, use_container_width=True)

                # 保存预测结果
                if st.button("💾 保存预测结果", type="secondary"):
                    try:
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT INTO ml_predictions 
                            (stock_code, model_name, prediction_date, prediction_days, 
                             mse, mae, r2, predictions)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            stock_code, model_type, datetime.now().date(), forecast_days,
                            float(metrics['mse']), float(metrics['mae']), float(metrics['r2']),
                            str(future_predictions_original.tolist())
                        ))
                        conn.commit()
                        st.success("✅ 预测结果已保存到数据库")
                    except Exception as e:
                        st.error(f"❌ 保存失败：{str(e)}")

                # 保存模型
                if st.button("💾 保存模型", type="secondary"):
                    model_path = f"models/{stock_code}_{model_type}_{datetime.now().strftime('%Y%m%d')}.h5"
                    import os
                    os.makedirs("models", exist_ok=True)

                    if model_type in ["LSTM", "LSTM with Attention", "BiLSTM"]:
                        model.save(model_path)
                    else:
                        import joblib
                        joblib.dump(model, model_path.replace('.h5', '.pkl'))

                    st.success(f"✅ 模型已保存到 {model_path}")

    # ========== 功能6：数据库管理 ==========
    elif func_choice == "数据库管理":
        st.subheader("🗃️ 数据库管理")

        try:
            # 显示统计信息
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM stock_data")
            total_records = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT stock_code) FROM stock_data")
            stock_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM ml_predictions")
            pred_count = cursor.fetchone()[0]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 总记录数", f"{total_records:,}")
            with col2:
                st.metric("📈 股票数量", stock_count)
            with col3:
                st.metric("🤖 预测记录", pred_count)

            # 显示预测历史
            if pred_count > 0:
                st.subheader("📋 预测历史记录")
                pred_history = pd.read_sql_query(
                    "SELECT * FROM ml_predictions ORDER BY created_time DESC LIMIT 10",
                    conn
                )
                st.dataframe(pred_history, use_container_width=True)

        except Exception as e:
            st.error(f"❌ 数据库管理错误：{str(e)}")

    # 侧边栏信息
    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### 📌 使用说明
    1. **数据抓取**：获取股票历史数据
    2. **数据显示**：可视化分析数据
    3. **数据存储**：保存数据到本地
    4. **数据更新**：增量更新数据库
    5. **股价预测**：机器学习预测股价走势
    6. **数据库管理**：管理本地数据
    """)

    # 底部信息
    st.markdown("---")
    st.caption(f"最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 关闭数据库连接
    conn.close()


if __name__ == "__main__":
    main()