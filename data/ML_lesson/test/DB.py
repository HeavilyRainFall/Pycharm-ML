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
from fake_useragent import UserAgent
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
import talib  # 新增：技术指标计算库

warnings.filterwarnings('ignore')


# ===================== 爬虫伪装配置增强 =====================
def setup_session_with_retry():
    """配置带重试和伪装头的requests session，增强反爬虫"""
    session = requests.Session()

    # 增强重试策略
    retry_strategy = Retry(
        total=5,  # 增加重试次数
        backoff_factor=2,  # 增加退避因子
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    # 更丰富的随机请求头
    ua = UserAgent()
    headers = {
        'User-Agent': ua.random,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
        'Referer': 'https://finance.sina.com.cn/',  # 增加 referer
        'DNT': '1'  # 增加 Do Not Track
    }
    session.headers.update(headers)

    return session


# ===================== 技术指标计算 =====================
def calculate_indicators(df):
    """计算常用技术指标"""
    if df is None or df.empty:
        return df

    df_copy = df.copy()

    # RSI指标 (相对强弱指数)
    df_copy['RSI14'] = talib.RSI(df_copy['close'], timeperiod=14)

    # MACD指标
    df_copy['MACD'], df_copy['MACD_signal'], df_copy['MACD_hist'] = talib.MACD(
        df_copy['close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # KDJ指标
    df_copy['K'], df_copy['D'] = talib.STOCH(
        df_copy['high'], df_copy['low'], df_copy['close'],
        fastk_period=9, slowk_period=3, slowk_matype=0,
        slowd_period=3, slowd_matype=0
    )
    df_copy['J'] = 3 * df_copy['K'] - 2 * df_copy['D']  # J线计算

    # 布林带
    df_copy['BB_upper'], df_copy['BB_middle'], df_copy['BB_lower'] = talib.BBANDS(
        df_copy['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

    return df_copy


# ===================== 数据库初始化 =====================
def init_db(db_path="quant_analysis.db"):
    """初始化SQLite数据库，创建股票数据表（支持技术指标）"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 扩展数据表结构，支持技术指标
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
        rsi14 FLOAT,
        macd FLOAT,
        macd_signal FLOAT,
        macd_hist FLOAT,
        k FLOAT,
        d FLOAT,
        j FLOAT,
        bb_upper FLOAT,
        bb_middle FLOAT,
        bb_lower FLOAT,
        freq TEXT DEFAULT 'daily',
        update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (stock_code, date, freq)
    )
    ''')

    # 创建索引提高查询速度
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_data(stock_code, date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON stock_data(date)')

    conn.commit()
    return conn


# ===================== 数据抓取（增强反爬虫） =====================
def get_stock_data(stock_code, start_date, end_date, freq="daily", indicators=None):
    """
    抓取A股股票数据（支持多股票、技术指标）
    增强反爬虫策略
    """
    try:
        # 更长的随机延时（2-5秒）
        time.sleep(random.uniform(2, 5))

        # 自动补全市场标识（沪市.SH/深市.SZ）
        if stock_code.startswith(('6', '9', '5')):
            market = "SH"
            stock_code_full = f"{stock_code}.SH"
        elif stock_code.startswith(('0', '3', '2')):
            market = "SZ"
            stock_code_full = f"{stock_code}.SZ"
        elif stock_code.startswith('8'):
            market = "BJ"
            stock_code_full = f"{stock_code}.BJ"
        else:
            st.error(f"暂不支持该市场代码 {stock_code}（仅支持沪深A股）")
            return None

        # 转换频率参数（akshare的格式）
        period_map = {
            "daily": "daily",
            "weekly": "weekly",
            "monthly": "monthly"
        }

        if freq not in period_map:
            freq = "daily"

        # 调用akshare获取数据（前复权）
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period=period_map[freq],
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"  # 前复权（更适合量化分析）
        )

        if df is None or df.empty:
            st.warning(f"⚠️ 未获取到{stock_code}在{start_date}到{end_date}的数据")
            return None

        # 数据标准化处理
        df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"
        }, inplace=True)

        # 添加额外信息
        df["stock_code"] = stock_code
        df["date"] = pd.to_datetime(df["date"]).dt.date  # 统一日期格式
        df["freq"] = freq

        # 转换数据类型
        numeric_cols = ["open", "close", "high", "low", "volume", "amount"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 计算技术指标（如果需要）
        if indicators and any(indicators):
            df = calculate_indicators(df)

        df.reset_index(drop=True, inplace=True)

        st.success(f"✅ 成功获取{stock_code} {freq}数据，共{len(df)}条记录")
        return df

    except Exception as e:
        st.error(f"❌ {stock_code}抓取失败：{str(e)}")
        # 更详细的错误信息
        import traceback
        st.error(f"详细错误：{traceback.format_exc()}")
        return None


# ===================== 批量数据抓取 =====================
def batch_get_stock_data(stock_codes, start_date, end_date, freq="daily", indicators=None):
    """批量获取多个股票数据"""
    all_data = []

    # 去重并处理股票代码
    code_list = [code.strip() for code in stock_codes.split(',') if code.strip()]
    unique_codes = list(set(code_list))  # 去重

    st.info(f"开始批量获取 {len(unique_codes)} 只股票数据...")

    for i, code in enumerate(unique_codes, 1):
        st.subheader(f"处理第 {i}/{len(unique_codes)} 只股票: {code}")
        df = get_stock_data(code, start_date, end_date, freq, indicators)

        if df is not None and not df.empty:
            all_data.append(df)
            # 股票间增加更长的随机间隔，降低反爬风险
            if i < len(unique_codes):
                sleep_time = random.uniform(3, 7)
                st.info(f"等待 {sleep_time:.1f} 秒后继续下一只股票...")
                time.sleep(sleep_time)

    if not all_data:
        st.warning("⚠️ 未获取到任何股票数据")
        return None

    # 合并所有股票数据
    combined_df = pd.concat(all_data, ignore_index=True)
    st.success(f"✅ 批量获取完成，共获取 {len(combined_df)} 条记录")
    return combined_df


# ===================== 数据显示（支持多股票和指标） =====================
def show_stock_data(df, stock_codes):
    """可视化展示股票数据（支持多股票和技术指标）"""
    if df is None or df.empty:
        st.warning("📭 暂无数据可显示")
        return

    # 1. 数据表格（可筛选）
    st.subheader(f"股票原始数据（含技术指标）")

    # 多股票筛选器
    unique_codes = df['stock_code'].unique()
    selected_code = st.selectbox("选择股票代码查看详情", unique_codes)
    filtered_df = df[df['stock_code'] == selected_code]

    st.dataframe(filtered_df, use_container_width=True)

    # 下载按钮
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 下载全部数据CSV",
        data=csv,
        file_name=f"multi_stocks_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

    # 2. 基本统计
    st.subheader("📊 数据统计摘要")
    stats = filtered_df[["open", "close", "high", "low", "volume", "amount"]].describe()
    st.write(stats)

    # 3. 收盘价走势
    st.subheader("📈 收盘价走势")
    if len(unique_codes) > 1:
        # 多股票对比
        fig_price = px.line(
            df, x="date", y="close", color="stock_code",
            title=f"多股票收盘价走势对比（前复权）",
            labels={"date": "日期", "close": "收盘价（元）", "stock_code": "股票代码"},
            template="plotly_white"
        )
    else:
        # 单股票走势
        fig_price = px.line(
            filtered_df, x="date", y="close",
            title=f"{selected_code} 收盘价走势（前复权）",
            labels={"date": "日期", "close": "收盘价（元）"},
            template="plotly_white"
        )

        # 添加移动平均线
        if len(filtered_df) > 20:
            filtered_df['MA20'] = filtered_df['close'].rolling(window=20).mean()
            filtered_df['MA60'] = filtered_df['close'].rolling(window=60).mean()

            fig_price.add_scatter(x=filtered_df['date'], y=filtered_df['MA20'],
                                  mode='lines', name='20日均线',
                                  line=dict(color='orange', width=2))
            fig_price.add_scatter(x=filtered_df['date'], y=filtered_df['MA60'],
                                  mode='lines', name='60日均线',
                                  line=dict(color='green', width=2))

    st.plotly_chart(fig_price, use_container_width=True)

    # 4. K线图（仅单股票）
    if len(unique_codes) == 1:
        st.subheader("📉 K线图")
        fig_kline = go.Figure(data=[go.Candlestick(
            x=filtered_df["date"],
            open=filtered_df["open"], high=filtered_df["high"],
            low=filtered_df["low"], close=filtered_df["close"],
            name="K线",
            increasing_line_color="#ff4b4b",  # 涨红
            decreasing_line_color="#009966"  # 跌绿
        )])

        # 添加成交量
        fig_kline.add_trace(go.Bar(
            x=filtered_df["date"],
            y=filtered_df["volume"],
            name="成交量",
            marker_color="rgba(100, 100, 200, 0.5)",
            yaxis="y2"
        ))

        fig_kline.update_layout(
            title=f"{selected_code} K线图（前复权）",
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

    # 5. 技术指标图（如果存在）
    if 'RSI14' in filtered_df.columns:
        st.subheader("📊 技术指标")

        # RSI指标图
        fig_rsi = px.line(
            filtered_df, x="date", y="RSI14",
            title=f"{selected_code} RSI指标",
            labels={"date": "日期", "RSI14": "RSI(14)"},
            template="plotly_white"
        )
        # 添加超买超卖线
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="超买线(70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="超卖线(30)")
        st.plotly_chart(fig_rsi, use_container_width=True)

        # MACD指标图
        if 'MACD' in filtered_df.columns:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=filtered_df["date"], y=filtered_df["MACD"],
                name="MACD", line=dict(color="blue")
            ))
            fig_macd.add_trace(go.Scatter(
                x=filtered_df["date"], y=filtered_df["MACD_signal"],
                name="信号线", line=dict(color="orange")
            ))
            fig_macd.add_trace(go.Bar(
                x=filtered_df["date"], y=filtered_df["MACD_hist"],
                name="MACD柱", marker_color="gray"
            ))
            fig_macd.update_layout(
                title=f"{selected_code} MACD指标",
                xaxis_title="日期",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_macd, use_container_width=True)


# ===================== 数据存储（支持多股票） =====================
def save_stock_data(df, save_type, db_conn=None):
    """存储多股票数据（支持CSV/Excel/SQLite）"""
    if df is None or df.empty:
        st.warning("📭 暂无数据可存储")
        return False

    try:
        # 确保data目录存在
        import os
        os.makedirs("data", exist_ok=True)

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multi_stocks_{timestamp}"

        # 1. 保存为CSV
        if save_type == "csv":
            filepath = f"data/{filename}.csv"
            df.to_csv(filepath, index=False, encoding="utf-8")
            st.success(f"💾 已保存为CSV：{filepath}")
            return True

        # 2. 保存为Excel
        elif save_type == "excel":
            filepath = f"data/{filename}.xlsx"
            df.to_excel(filepath, index=False, engine="openpyxl")
            st.success(f"💾 已保存为Excel：{filepath}")
            return True

        # 3. 存入SQLite
        elif save_type == "sqlite":
            if not db_conn:
                st.error("❌ SQLite连接未初始化")
                return False

            # 按股票代码和频率分批处理
            stock_codes = df['stock_code'].unique()
            freq_list = df['freq'].unique() if 'freq' in df.columns else ['daily']

            for code in stock_codes:
                for freq in freq_list:
                    code_freq_df = df[(df['stock_code'] == code) & (df['freq'] == freq)]

                    # 删除该股票该频率的旧数据，然后插入新数据
                    cursor = db_conn.cursor()
                    cursor.execute(
                        "DELETE FROM stock_data WHERE stock_code = ? AND freq = ?",
                        (code, freq)
                    )
                    db_conn.commit()

                    # 插入新数据
                    code_freq_df.to_sql("stock_data", db_conn, if_exists="append", index=False)

            st.success(f"💾 已存入SQLite数据库，共{len(df)}条记录，涉及{len(stock_codes)}只股票")
            return True

        else:
            st.error("❌ 存储类型错误（仅支持csv/excel/sqlite）")
            return False

    except Exception as e:
        st.error(f"❌ 存储失败：{str(e)}")
        import traceback
        st.error(f"详细错误：{traceback.format_exc()}")
        return False


# ===================== 数据更新（支持多股票） =====================
def update_stock_data(stock_code, freq, db_conn):
    """增量更新数据库中的股票数据（仅更新最新日期后的数据）"""
    try:
        cursor = db_conn.cursor()

        # 1. 查询数据库中该股票该频率的最新日期
        cursor.execute(
            """
            SELECT MAX(date) 
            FROM stock_data 
            WHERE stock_code = ? AND freq = ?
            """,
            (stock_code, freq)
        )
        result = cursor.fetchone()
        latest_date = result[0] if result and result[0] else None

        if not latest_date:
            st.warning(f"📭 数据库无{stock_code}的{freq}数据，请先抓取并存储")
            return None

        # 2. 计算更新时间范围
        latest_date = datetime.strptime(latest_date, "%Y-%m-%d").date()
        start_date = (latest_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        # 如果开始日期大于结束日期，说明数据已是最新
        if start_date > end_date:
            st.success(f"✅ {stock_code}的{freq}数据已是最新，无需更新")
            # 返回现有数据
            df = pd.read_sql_query(
                f"SELECT * FROM stock_data WHERE stock_code = '{stock_code}' AND freq = '{freq}' ORDER BY date",
                db_conn
            )
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df

        # 3. 抓取增量数据（包含已选技术指标）
        st.info(f"🔄 开始更新{stock_code}的{freq}数据（{start_date} ~ {end_date}）")
        # 检查是否有技术指标列
        cursor.execute("PRAGMA table_info(stock_data)")
        columns = [col[1] for col in cursor.fetchall()]
        has_indicators = 'rsi14' in columns  # 检查是否有技术指标列

        df_new = get_stock_data(stock_code, start_date, end_date, freq, indicators=has_indicators)

        if df_new is None or df_new.empty:
            st.warning(f"📭 {stock_code}无{start_date}到{end_date}的增量数据")
            df = pd.read_sql_query(
                f"SELECT * FROM stock_data WHERE stock_code = '{stock_code}' AND freq = '{freq}' ORDER BY date",
                db_conn
            )
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df

        # 4. 保存增量数据到数据库
        success = save_stock_data(df_new, "sqlite", db_conn)

        if success:
            # 5. 返回更新后的完整数据
            df_updated = pd.read_sql_query(
                f"SELECT * FROM stock_data WHERE stock_code = '{stock_code}' AND freq = '{freq}' ORDER BY date",
                db_conn
            )
            df_updated["date"] = pd.to_datetime(df_updated["date"]).dt.date
            return df_updated
        else:
            return None

    except Exception as e:
        st.error(f"❌ 更新失败：{str(e)}")
        import traceback
        st.error(f"详细错误：{traceback.format_exc()}")
        return None


# ===================== 主界面（Streamlit） =====================
def main():
    # 页面基础配置
    st.set_page_config(
        page_title="量化分析系统 - 数据获取",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 页面标题
    st.title("📈 量化分析系统 | 数据获取模块")
    st.markdown("---")

    # 初始化数据库连接
    conn = init_db()

    # 侧边栏功能菜单
    st.sidebar.title("⚙️ 功能菜单")
    func_choice = st.sidebar.radio(
        "选择功能",
        ["数据抓取", "数据显示", "数据存储", "数据更新", "数据库管理"],
        index=0
    )

    # 会话状态：缓存抓取的数据和股票代码
    if "stock_data" not in st.session_state:
        st.session_state["stock_data"] = None
    if "stock_codes" not in st.session_state:
        st.session_state["stock_codes"] = "600000,000001"  # 多股票默认值
    if "freq" not in st.session_state:
        st.session_state["freq"] = "daily"
    if "indicators" not in st.session_state:
        st.session_state["indicators"] = ["RSI", "MACD", "KDJ", "布林带"]  # 默认技术指标

    # ========== 功能1：数据抓取 ==========
    if func_choice == "数据抓取":
        st.subheader("🔍 股票数据抓取（支持多股票）")

        col1, col2 = st.columns([1, 2])
        with col1:
            stock_codes = st.text_input(
                "股票代码（逗号分隔）",
                value="600000,000001",
                help="如600000,000001,000300（多个代码用逗号分隔）"
            )
            freq = st.selectbox(
                "数据频率",
                ["daily", "weekly", "monthly"],
                index=0,
                help="日线/周线/月线"
            )

            # 技术指标选择
            st.subheader("选择技术指标")
            indicators = []
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                rsi = st.checkbox("RSI（相对强弱指数）", value=True)
                macd = st.checkbox("MACD（指数平滑异同平均线）", value=True)
            with col1_2:
                kdj = st.checkbox("KDJ（随机指标）", value=True)
                boll = st.checkbox("布林带", value=True)

            if rsi:
                indicators.append("RSI")
            if macd:
                indicators.append("MACD")
            if kdj:
                indicators.append("KDJ")
            if boll:
                indicators.append("布林带")

        with col2:
            date_range = st.date_input(
                "选择日期范围",
                value=[datetime.now() - timedelta(days=180), datetime.now()],
                max_value=datetime.now()
            )

            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = datetime.now() - timedelta(days=180)
                end_date = datetime.now()

        if st.button("🚀 开始抓取", type="primary", use_container_width=True):
            with st.spinner(f"正在抓取股票数据..."):
                # 转换日期格式
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")

                # 执行批量抓取并缓存
                df = batch_get_stock_data(stock_codes, start_str, end_str, freq, indicators)

                if df is not None:
                    st.session_state["stock_data"] = df
                    st.session_state["stock_codes"] = stock_codes
                    st.session_state["freq"] = freq
                    st.session_state["indicators"] = indicators

                    # 显示前5行数据
                    st.subheader("📋 数据预览")
                    st.dataframe(df.head(), use_container_width=True)

    # ========== 功能2：数据显示 ==========
    elif func_choice == "数据显示":
        st.subheader("📊 数据可视化展示")

        if st.session_state["stock_data"] is None:
            st.warning("📭 请先抓取数据！")

            # 快速抓取示例数据
            if st.button("📌 快速抓取示例（多股票近90天）"):
                start_str = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                end_str = datetime.now().strftime("%Y-%m-%d")
                df = batch_get_stock_data("600000,000001", start_str, end_str, "daily", ["RSI", "MACD"])
                st.session_state["stock_data"] = df
                st.session_state["stock_codes"] = "600000,000001"
                st.rerun()
        else:
            show_stock_data(
                st.session_state["stock_data"],
                st.session_state["stock_codes"]
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

            col1, col2 = st.columns([3, 1])
            with col1:
                unique_codes = st.session_state["stock_data"]['stock_code'].unique()
                st.info(f"将存储 {len(unique_codes)} 只股票的 {len(st.session_state['stock_data'])} 条记录")

            with col2:
                if st.button("💾 开始存储", type="primary", use_container_width=True):
                    with st.spinner("正在存储数据..."):
                        success = save_stock_data(
                            st.session_state["stock_data"],
                            save_type,
                            conn
                        )

                        if success and save_type == "sqlite":
                            # 显示数据库中的股票列表
                            st.subheader("📂 数据库中的股票列表")
                            stock_list = pd.read_sql_query(
                                "SELECT DISTINCT stock_code, freq, COUNT(*) as record_count, MAX(date) as latest_date FROM stock_data GROUP BY stock_code, freq",
                                conn
                            )
                            st.dataframe(stock_list, use_container_width=True)

    # ========== 功能4：数据更新 ==========
    elif func_choice == "数据更新":
        st.subheader("🔄 增量更新数据")

        col1, col2 = st.columns(2)
        with col1:
            stock_code = st.text_input(
                "待更新股票代码（单个）",
                value="600000"
            )

        with col2:
            freq = st.selectbox(
                "数据频率",
                ["daily", "weekly", "monthly"],
                index=["daily", "weekly", "monthly"].index(st.session_state.get("freq", "daily"))
            )

        if st.button("🔄 开始更新", type="primary", use_container_width=True):
            with st.spinner(f"正在更新{stock_code}的{freq}数据..."):
                updated_df = update_stock_data(stock_code, freq, conn)
                if updated_df is not None:
                    st.session_state["stock_data"] = updated_df
                    st.session_state["stock_codes"] = stock_code
                    st.session_state["freq"] = freq
                    st.success("更新完成！")
                    st.dataframe(updated_df.tail(), use_container_width=True)

    # ========== 功能5：数据库管理 ==========
    elif func_choice == "数据库管理":
        st.subheader("🗄️ 数据库管理")

        # 显示数据库中的股票
        st.subheader("数据库中的股票数据")
        stock_list = pd.read_sql_query(
            "SELECT DISTINCT stock_code, freq, COUNT(*) as record_count, MAX(date) as latest_date FROM stock_data GROUP BY stock_code, freq",
            conn
        )
        st.dataframe(stock_list, use_container_width=True)

        # 批量更新选项
        if not stock_list.empty:
            st.subheader("批量更新")
            all_codes = stock_list['stock_code'].unique()
            selected_codes = st.multiselect("选择要更新的股票", all_codes, default=all_codes[:3])

            if st.button("批量更新选中股票", type="primary"):
                for code in selected_codes:
                    st.subheader(f"更新 {code} ...")
                    freqs = stock_list[stock_list['stock_code'] == code]['freq'].unique()
                    for freq in freqs:
                        update_stock_data(code, freq, conn)
                st.success("批量更新完成！")

        # 清空数据选项
        with st.expander("危险操作：清空数据", expanded=False):
            del_code = st.text_input("输入要删除的股票代码（全部删除请输入'all'）")
            if st.button("确认删除"):
                if del_code == "all":
                    confirm = st.checkbox("我确认要删除所有数据！")
                    if confirm:
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM stock_data")
                        conn.commit()
                        st.success("已删除所有数据")
                elif del_code:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM stock_data WHERE stock_code = ?", (del_code,))
                    conn.commit()
                    st.success(f"已删除 {del_code} 的所有数据")

    # 关闭数据库连接
    conn.close()


if __name__ == "__main__":
    main()