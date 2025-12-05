# DB_SA.py - 升级版：支持中文显示 + 多种可调机器学习模型预测股价

import os
import sqlite3
import pandas as pd
import numpy as np
import akshare as ak
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==============================
# 🕵️ 爬虫伪装
# ==============================
os.environ["AKSHARE_HEADERS"] = str({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.baidu.com"
})

# ==============================
# 🖋️ 修复 Plotly 中文显示（关键！）
# ==============================
import plotly.io as pio
pio.templates.default = "plotly_white"
# 设置全局字体（支持中文）
px.defaults.font_family = "SimHei, Microsoft YaHei, sans-serif"
go.layout.Template().layout.font.family = "SimHei, Microsoft YaHei, sans-serif"

# ==============================
# 🗃️ 数据库初始化
# ==============================
def init_db(db_path="stock_data.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            stock_code TEXT,
            date TEXT,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume REAL,
            amount REAL,
            PRIMARY KEY (stock_code, date)
        )
    """)
    conn.commit()
    return conn

# ==============================
# 📥 获取股票数据
# ==============================
def get_stock_data(stock_code: str, start_date: str, end_date: str):
    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
            adjust="qfq"
        )
        if df.empty:
            return None
        df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"
        }, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["stock_code"] = stock_code
        return df[["stock_code", "date", "open", "close", "high", "low", "volume", "amount"]]
    except Exception as e:
        st.error(f"❌ 抓取失败: {e}")
        return None

# ==============================
# 💾 保存到 SQLite（防重复）
# ==============================
def save_to_db(df, conn):
    if df is None or df.empty:
        return
    cursor = conn.cursor()
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR IGNORE INTO stock_data 
            (stock_code, date, open, close, high, low, volume, amount)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, tuple(row))
    conn.commit()

# ==============================
# 🔁 更新数据
# ==============================
def update_stock_data(stock_code, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(date) FROM stock_data WHERE stock_code = ?", (stock_code,))
    last = cursor.fetchone()[0]
    start_str = (datetime.strptime(last, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y%m%d") if last else "20200101"
    end_str = datetime.today().strftime("%Y%m%d")
    if start_str > end_str:
        st.info("✅ 已是最新数据")
        return
    df_new = get_stock_data(stock_code, start_str[:4] + "-" + start_str[4:6] + "-" + start_str[6:], end_str[:4] + "-" + end_str[4:6] + "-" + end_str[6:])
    if df_new is not None and not df_new.empty:
        save_to_db(df_new, conn)
        st.success(f"✅ 新增 {len(df_new)} 条数据")

# ==============================
# 🧠 构建 Attention 层（用于 LSTM+Attention）
# ==============================
from tensorflow.keras.layers import Layer
import tensorflow as tf

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        # inputs: (batch, timesteps, features)
        e = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1))  # (batch, timesteps, 1)
        a = tf.nn.softmax(e, axis=1)  # (batch, timesteps, 1)
        output = inputs * a  # (batch, timesteps, features)
        return tf.reduce_sum(output, axis=1)  # (batch, features)

# ==============================
# 🤖 创建模型（支持多种架构）
# ==============================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def create_model(model_type, seq_length, n_features, lstm_units=50, use_attention=False, dropout=0.2):
    model = Sequential()
    if model_type == "LSTM":
        if use_attention:
            model.add(LSTM(lstm_units, return_sequences=True, input_shape=(seq_length, n_features)))
            model.add(Attention())
        else:
            model.add(LSTM(lstm_units, input_shape=(seq_length, n_features)))
    elif model_type == "GRU":
        model.add(GRU(lstm_units, input_shape=(seq_length, n_features)))
    elif model_type == "SimpleRNN":
        model.add(SimpleRNN(lstm_units, input_shape=(seq_length, n_features)))
    else:
        raise ValueError("Unsupported model type")

    model.add(Dropout(dropout))
    model.add(Dense(1))
    return model

# ==============================
# 📊 训练与预测
# ==============================
@st.cache_resource
def train_and_predict(stock_code, df, model_type, seq_length, epochs, lstm_units, learning_rate, use_attention, dropout):
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    data = df[['close']].values.astype(float)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # 构造序列
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = create_model(model_type, seq_length, 1, lstm_units, use_attention, dropout)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )

    # 预测
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

    return history, train_pred, test_pred, y_train_inv, y_test_inv, df['date'].iloc[seq_length:split+seq_length], df['date'].iloc[split+seq_length:]

# ==============================
# 🖼️ 主界面
# ==============================
def main():
    st.set_page_config(page_title="📈 量化分析系统", layout="wide")
    st.title("📈 量化分析系统 | 数据 + 预测")
    st.divider()

    conn = init_db()

    # 会话状态
    if "stock_code" not in st.session_state:
        st.session_state["stock_code"] = "600000"
    if "df" not in st.session_state:
        st.session_state["df"] = None

    # 侧边栏
    func = st.sidebar.radio("功能", ["数据抓取", "数据显示", "股价预测"])

    # ========== 数据抓取 ==========
    if func == "数据抓取":
        st.subheader("🔍 抓取股票数据")
        code = st.text_input("股票代码", value=st.session_state["stock_code"])
        start = st.date_input("开始日期", datetime.today() - timedelta(days=365))
        end = st.date_input("结束日期", datetime.today())
        if st.button("🚀 抓取"):
            df = get_stock_data(code, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
            if df is not None:
                save_to_db(df, conn)
                st.session_state["df"] = df
                st.session_state["stock_code"] = code
                st.success("✅ 数据已保存")

    # ========== 数据显示 ==========
    elif func == "数据显示":
        st.subheader("📊 历史数据展示")
        code = st.text_input("股票代码", value=st.session_state["stock_code"])
        if st.button("🔄 加载数据"):
            df = pd.read_sql("SELECT * FROM stock_data WHERE stock_code = ? ORDER BY date", conn, params=(code,))
            if not df.empty:
                st.session_state["df"] = df
                st.session_state["stock_code"] = code
            else:
                st.warning("⚠️ 无数据，请先抓取")

        if st.session_state["df"] is not None:
            df = st.session_state["df"]
            st.dataframe(df)

            fig = px.line(df, x='date', y='close', title=f"{code} 收盘价走势")
            st.plotly_chart(fig, use_container_width=True)

    # ========== 股价预测 ==========
    elif func == "股价预测":
        st.subheader("🤖 股价预测（机器学习）")
        code = st.text_input("股票代码", value=st.session_state["stock_code"])
        df = pd.read_sql("SELECT * FROM stock_data WHERE stock_code = ? ORDER BY date", conn, params=(code,))
        if df.empty:
            st.warning("⚠️ 请先抓取该股票数据")
            return

        # 模型参数
        st.sidebar.subheader("⚙️ 模型参数")
        model_type = st.sidebar.selectbox("模型类型", ["LSTM", "LSTM + Attention", "GRU", "SimpleRNN"])
        use_attention = model_type == "LSTM + Attention"
        if use_attention:
            model_type = "LSTM"

        seq_length = st.sidebar.slider("序列长度（lookback）", 10, 100, 60)
        epochs = st.sidebar.slider("训练轮数（epochs）", 10, 100, 30)
        lstm_units = st.sidebar.slider("LSTM单元数", 16, 128, 50)
        learning_rate = st.sidebar.number_input("学习率", 0.0001, 0.01, 0.001, format="%.4f")
        dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2)

        if st.button("🧠 开始训练与预测"):
            with st.spinner("训练中..."):
                try:
                    history, train_pred, test_pred, y_train, y_test, train_dates, test_dates = train_and_predict(
                        code, df, model_type, seq_length, epochs, lstm_units, learning_rate, use_attention, dropout
                    )

                    # 损失曲线
                    fig_loss = px.line(
                        x=range(1, len(history.history['loss'])+1),
                        y=history.history['loss'],
                        labels={'x': 'Epoch', 'y': 'Loss'},
                        title="训练损失曲线"
                    )
                    st.plotly_chart(fig_loss, use_container_width=True)

                    # 预测结果
                    pred_df = pd.DataFrame({
                        'date': list(train_dates) + list(test_dates),
                        'actual': list(y_train.flatten()) + list(y_test.flatten()),
                        'predicted': list(train_pred.flatten()) + list(test_pred.flatten()),
                        'type': ['train'] * len(train_pred) + ['test'] * len(test_pred)
                    })

                    fig_pred = px.line(pred_df, x='date', y='predicted', color='type', title="股价预测 vs 实际")
                    fig_pred.add_scatter(x=pred_df['date'], y=pred_df['actual'], mode='lines', name='实际价格')
                    st.plotly_chart(fig_pred, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ 训练失败: {e}")

    conn.close()

if __name__ == "__main__":
    main()