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

warnings.filterwarnings('ignore')


# ===================== 爬虫伪装配置 =====================
def setup_session_with_retry():
    """配置带重试和伪装头的requests session"""
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

    # 随机请求头
    ua = UserAgent()
    headers = {
        'User-Agent': ua.random,
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

    # 创建索引提高查询速度
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_data(stock_code, date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON stock_data(date)')

    conn.commit()
    return conn


# ===================== 数据抓取（带伪装） =====================
def get_stock_data(stock_code, start_date, end_date, freq="daily"):
    """
    抓取A股股票数据（支持日线/周线/月线，前复权）
    添加随机延时和请求头伪装
    """
    try:
        # 随机延时（1-3秒）
        time.sleep(random.uniform(1, 3))

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
            st.error("暂不支持该市场代码（仅支持沪深A股）")
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

        df.reset_index(drop=True, inplace=True)

        st.success(f"✅ 成功获取{stock_code} {freq}数据，共{len(df)}条记录")
        return df

    except Exception as e:
        st.error(f"❌ 抓取失败：{str(e)}")
        # 更详细的错误信息
        import traceback
        st.error(f"详细错误：{traceback.format_exc()}")
        return None


# ===================== 数据显示 =====================
def show_stock_data(df, stock_code):
    """可视化展示股票数据（表格+统计+走势+K线）"""
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
    if len(df) > 20:
        df_copy = df.copy()
        df_copy['MA20'] = df_copy['close'].rolling(window=20).mean()
        df_copy['MA60'] = df_copy['close'].rolling(window=60).mean()

        fig_price.add_scatter(x=df_copy['date'], y=df_copy['MA20'],
                              mode='lines', name='20日均线',
                              line=dict(color='orange', width=2))
        fig_price.add_scatter(x=df_copy['date'], y=df_copy['MA60'],
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
        increasing_line_color="#ff4b4b",  # 涨红
        decreasing_line_color="#009966"  # 跌绿
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


# ===================== 数据存储 =====================
def save_stock_data(df, stock_code, save_type, db_conn=None):
    """
    存储数据（支持CSV/Excel/SQLite）
    :param save_type: 存储类型（csv/excel/sqlite）
    :return: bool 存储是否成功
    """
    if df is None or df.empty:
        st.warning("📭 暂无数据可存储")
        return False

    try:
        # 确保data目录存在
        import os
        os.makedirs("data", exist_ok=True)

        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stock_code}_stock_{timestamp}"

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

            # 首先删除该股票该频率的旧数据，然后插入新数据
            cursor = db_conn.cursor()

            # 获取频率（默认为daily）
            freq = df.iloc[0]['freq'] if 'freq' in df.columns else 'daily'

            # 删除该股票该频率的旧数据（避免重复）
            cursor.execute(
                "DELETE FROM stock_data WHERE stock_code = ? AND freq = ?",
                (stock_code, freq)
            )
            db_conn.commit()

            # 插入新数据
            df.to_sql("stock_data", db_conn, if_exists="append", index=False)
            st.success(f"💾 已存入SQLite数据库，共{len(df)}条记录")
            return True

        else:
            st.error("❌ 存储类型错误（仅支持csv/excel/sqlite）")
            return False

    except Exception as e:
        st.error(f"❌ 存储失败：{str(e)}")
        import traceback
        st.error(f"详细错误：{traceback.format_exc()}")
        return False


# ===================== 数据更新 =====================
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

        # 3. 抓取增量数据
        st.info(f"🔄 开始更新{stock_code}的{freq}数据（{start_date} ~ {end_date}）")
        df_new = get_stock_data(stock_code, start_date, end_date, freq)

        if df_new is None or df_new.empty:
            st.warning(f"📭 {stock_code}无{start_date}到{end_date}的增量数据")
            df = pd.read_sql_query(
                f"SELECT * FROM stock_data WHERE stock_code = '{stock_code}' AND freq = '{freq}' ORDER BY date",
                db_conn
            )
            df["date"] = pd.to_datetime(df["date"]).dt.date
            return df

        # 4. 保存增量数据到数据库
        success = save_stock_data(df_new, stock_code, "sqlite", db_conn)

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
    if "stock_code" not in st.session_state:
        st.session_state["stock_code"] = "600000"
    if "freq" not in st.session_state:
        st.session_state["freq"] = "daily"

    # ========== 功能1：数据抓取 ==========
    if func_choice == "数据抓取":
        st.subheader("🔍 股票数据抓取")

        col1, col2 = st.columns([1, 2])
        with col1:
            stock_code = st.text_input(
                "股票代码",
                value="600000",
                help="如600000（浦发银行）、000001（平安银行）、000300（沪深300）"
            )
            freq = st.selectbox(
                "数据频率",
                ["daily", "weekly", "monthly"],
                index=0,
                help="日线/周线/月线"
            )

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
            with st.spinner(f"正在抓取{stock_code}的{freq}数据..."):
                # 转换日期格式
                start_str = start_date.strftime("%Y-%m-%d")
                end_str = end_date.strftime("%Y-%m-%d")

                # 执行抓取并缓存
                df = get_stock_data(stock_code, start_str, end_str, freq)

                if df is not None:
                    st.session_state["stock_data"] = df
                    st.session_state["stock_code"] = stock_code
                    st.session_state["freq"] = freq

                    # 显示前5行数据
                    st.subheader("📋 数据预览")
                    st.dataframe(df.head(), use_container_width=True)

    # ========== 功能2：数据显示 ==========
    elif func_choice == "数据显示":
        st.subheader("📊 数据可视化展示")

        if st.session_state["stock_data"] is None:
            st.warning("📭 请先抓取数据！")

            # 快速抓取示例数据
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📌 快速抓取示例（600000近90天）"):
                    start_str = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
                    end_str = datetime.now().strftime("%Y-%m-%d")
                    df = get_stock_data("600000", start_str, end_str, "daily")
                    st.session_state["stock_data"] = df
                    st.session_state["stock_code"] = "600000"
                    st.rerun()

            with col2:
                if st.button("📌 快速抓取示例（000001近180天）"):
                    start_str = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
                    end_str = datetime.now().strftime("%Y-%m-%d")
                    df = get_stock_data("000001", start_str, end_str, "daily")
                    st.session_state["stock_data"] = df
                    st.session_state["stock_code"] = "000001"
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

            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"将存储 {st.session_state['stock_code']} 的 {len(st.session_state['stock_data'])} 条记录")

            with col2:
                if st.button("💾 开始存储", type="primary", use_container_width=True):
                    with st.spinner("正在存储数据..."):
                        success = save_stock_data(
                            st.session_state["stock_data"],
                            st.session_state["stock_code"],
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
                "待更新股票代码",
                value=st.session_state["stock_code"]
            )

        with col2:
            freq = st.selectbox(
                "数据频率",
                ["daily", "weekly", "monthly"],
                index=["daily", "weekly", "monthly"].index(st.session_state.get("freq", "daily"))
            )

        if st.button("🔄 开始更新", type="primary", use_container_width=True):
            with st.spinner(f"正在更新{stock_code}的{freq}数据..."):
                df_updated = update_stock_data(stock_code, freq, conn)

                if df_updated is not None:
                    st.session_state["stock_data"] = df_updated
                    st.session_state["stock_code"] = stock_code
                    st.session_state["freq"] = freq

                    # 显示更新后的数据
                    show_stock_data(df_updated, stock_code)

    # ========== 功能5：数据库管理 ==========
    elif func_choice == "数据库管理":
        st.subheader("🗃️ 数据库管理")

        try:
            # 显示数据库信息
            cursor = conn.cursor()

            # 1. 统计信息
            cursor.execute("SELECT COUNT(*) as total_records FROM stock_data")
            total_records = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT stock_code) as stock_count FROM stock_data")
            stock_count = cursor.fetchone()[0]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("📊 总记录数", f"{total_records:,}")
            with col2:
                st.metric("📈 股票数量", stock_count)

            # 2. 股票列表
            st.subheader("📋 数据库中的股票列表")
            stock_list = pd.read_sql_query(
                """
                SELECT 
                    stock_code,
                    freq,
                    COUNT(*) as record_count,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    MAX(update_time) as last_updated
                FROM stock_data 
                GROUP BY stock_code, freq
                ORDER BY stock_code, freq
                """,
                conn
            )
            st.dataframe(stock_list, use_container_width=True)

            # 3. 数据清理选项
            st.subheader("🧹 数据清理")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🗑️ 清空数据库", type="secondary"):
                    cursor.execute("DELETE FROM stock_data")
                    conn.commit()
                    st.success("✅ 数据库已清空")
                    st.rerun()

            with col2:
                stock_to_delete = st.text_input("删除指定股票数据", placeholder="输入股票代码")
                if st.button("🗑️ 删除该股票数据"):
                    if stock_to_delete:
                        cursor.execute("DELETE FROM stock_data WHERE stock_code = ?", (stock_to_delete,))
                        conn.commit()
                        st.success(f"✅ 已删除股票 {stock_to_delete} 的所有数据")
                        st.rerun()

            # 4. 数据库备份
            st.subheader("💾 数据库备份")
            if st.button("📤 导出数据库为CSV"):
                all_data = pd.read_sql_query("SELECT * FROM stock_data", conn)
                csv = all_data.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="📥 下载CSV备份",
                    data=csv,
                    file_name=f"stock_database_backup_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ 数据库管理错误：{str(e)}")

    # 侧边栏：关于信息
    st.sidebar.markdown("---")
    st.sidebar.info("""
    ### 📌 使用说明
    1. **数据抓取**：获取股票历史数据
    2. **数据显示**：可视化分析数据
    3. **数据存储**：保存数据到本地
    4. **数据更新**：增量更新数据库
    5. **数据库管理**：管理本地数据

    ### ⚠️ 注意事项
    - 数据来源：Akshare（免费接口）
    - 数据延迟：15分钟
    - 建议频率：日线数据
    """)

    # 底部信息
    st.markdown("---")
    st.caption(f"最后更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 关闭数据库连接
    conn.close()


if __name__ == "__main__":
    main()