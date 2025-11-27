import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
from dateutil import parser as date_parser
from concurrent.futures import ThreadPoolExecutor
import json
import re
import time

# =====================================================
# 0. 核心配置
# =====================================================

# ⚠️ 已删除硬编码的 Key，请在侧边栏配置
YOUR_GEMINI_KEY = "" 

DEFAULT_FASTFOREX_API_KEY = "639a184439-5da8887ddf-t6a0ml"
DEFAULT_ITICK_TOKEN = "d3e0fc48cacc405e92f4940f42ac171c88eb8cafb9e7418b88003cb098792c24"
DEFAULT_FMP_KEY = "o38qVA6O1AMMjlSV60PB22TTDaB97FX1"
DEFAULT_MARKETAUX_TOKEN = "gWBZhghNhnRYZnowqU8H4VKP35qEkXy1eXLRxgVU"

SYMBOL = "EURUSD"

try:
    from google import genai
    HAS_NEW_SDK = True
except ImportError:
    HAS_NEW_SDK = False

st.set_page_config(
    page_title="EUR/USD Pro v42.1 (Custom)",
    page_icon="🌗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# I. 极简 CSS (只做布局，不做颜色强制)
# =====================================================

def inject_layout_css():
    st.markdown("""
    <style>
        /* 新闻卡片边框装饰 (只定义边框颜色，不定义背景，适应亮/暗模式) */
        .news-box {
            border-left-width: 5px;
            border-left-style: solid;
            padding: 10px 15px;
            margin-bottom: 10px;
            /* 使用透明背景，让 Streamlit 自动决定底色 */
            background-color: rgba(128, 128, 128, 0.1); 
            border-radius: 0 5px 5px 0;
        }
        .border-red { border-left-color: #FF4B4B; }
        .border-orange { border-left-color: #FFA500; }
        .border-blue { border-left-color: #00BFFF; }
        
        .news-meta {
            font-size: 0.85rem;
            opacity: 0.8;
            margin-bottom: 4px;
        }
        
        .news-title {
            font-size: 1.1rem;
            font-weight: 600;
        }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# II. 数据抓取逻辑 (保持稳定)
# =====================================================

class NewsFetcher:
    @staticmethod
    def fetch_google_cn():
        items = []
        try:
            query = "欧元+OR+美元+OR+美联储+OR+汇率+OR+央行+OR+非农+OR+CPI"
            url = f"https://news.google.com/rss/search?q={query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers, timeout=5)
            root = ET.fromstring(r.content)
            for item in root.findall('./channel/item'):
                dt = date_parser.parse(item.find('pubDate').text)
                items.append({"title": item.find('title').text, "url": item.find('link').text, "source": "Google/CN", "lang": "zh", "dt": dt})
        except: pass
        return items

    @staticmethod
    def fetch_fmp(key):
        items = []
        if not key: return []
        try:
            url = f"https://financialmodelingprep.com/api/v4/forex_news?symbol=EURUSD&limit=30&apikey={key}"
            r = requests.get(url, timeout=5)
            for i in r.json():
                dt = date_parser.parse(i['publishedDate'])
                items.append({"title": i['title'], "url": i.get('link', '#'), "source": "FMP", "lang": "en", "dt": dt})
        except: pass
        return items

    @staticmethod
    def fetch_marketaux(token):
        items = []
        if not token: return []
        try:
            url = "https://api.marketaux.com/v1/news/all"
            params = {"api_token": token, "search": 'EURUSD OR "Fed" OR "ECB"', "language": "en", "limit": 30}
            r = requests.get(url, params=params, timeout=6)
            if 'data' in r.json():
                for i in r.json()['data']:
                    dt = date_parser.parse(i['published_at'])
                    items.append({"title": i['title'], "url": i['url'], "source": "MarketAux", "lang": "en", "dt": dt})
        except: pass
        return items

def ai_score_news(api_key, raw_news_list):
    if not raw_news_list: return []
    if not HAS_NEW_SDK: return raw_news_list[:5]
    # 如果没有 Key，直接返回部分新闻，不做评分
    if not api_key: return raw_news_list[:5]

    client = genai.Client(api_key=api_key)
    
    payload = "".join([f"ID {i}: {n['title']}\n" for i, n in enumerate(raw_news_list)])
    
    prompt = f"""
    You are a High-Frequency Trader. Assign Relevance Score (0-100) for EUR/USD.
    Criteria: 90+ (War/Rates), 75+ (Major Data), <50 (Noise/Stocks/Crypto).
    Input:\n{payload}
    Output JSON: [ {{"id": 0, "score": 95, "zh_title": "中文标题"}}, ... ]
    """
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text_res = response.text.strip()
        match = re.search(r'\[.*\]', text_res, re.DOTALL)
        high_value_items = []
        if match:
            scores = json.loads(match.group(0))
            for s in scores:
                idx = s.get('id')
                score = s.get('score', 0)
                if score >= 75 and idx < len(raw_news_list):
                    item = raw_news_list[idx]
                    item['title_zh'] = s.get('zh_title', item['title'])
                    item['ai_score'] = score
                    high_value_items.append(item)
            high_value_items.sort(key=lambda x: x['ai_score'], reverse=True)
            return high_value_items
        return []
    except: return []

def get_full_data(fast_key, itick_token, fmp_key, ma_token, k_type, ai_key):
    price = 0
    kline = []
    
    # 使用 ThreadPoolExecutor 并行请求
    with ThreadPoolExecutor(max_workers=5) as ex:
        t_price = ex.submit(requests.get, f"https://api.fastforex.io/fetch-one?from=EUR&to=USD&api_key={fast_key}")
        t_kline = ex.submit(requests.get, f"https://api.itick.org/forex/kline?region=GB&code={SYMBOL}&kType={k_type}&limit=200", headers={"token": itick_token})
        t_cn = ex.submit(NewsFetcher.fetch_google_cn)
        t_fmp = ex.submit(NewsFetcher.fetch_fmp, fmp_key)
        t_ma = ex.submit(NewsFetcher.fetch_marketaux, ma_token)
        
        # 获取实时价格 (FastForex 通常比 K线数据更实时)
        try: 
            price_res = t_price.result().json()
            if 'result' in price_res and 'EURUSD' in price_res['result']:
                price = float(price_res['result']['EURUSD'])
        except: 
            price = 0

        # 获取K线
        try: 
            r = t_kline.result().json()
            if r.get('code') == 0: 
                kline = r.get('data', [])
        except: pass
        
        # 获取新闻
        try: raw_news = t_cn.result() + t_fmp.result() + t_ma.result()
        except: raw_news = []

    # 新闻去重与筛选逻辑
    now = datetime.now(timezone.utc)
    candidates = []
    seen = set()
    raw_news.sort(key=lambda x: x['dt'], reverse=True)
    
    for n in raw_news:
        if n['dt'].tzinfo is None: n['dt'] = n['dt'].replace(tzinfo=timezone.utc)
        else: n['dt'] = n['dt'].astimezone(timezone.utc)
        
        if (now - n['dt']).total_seconds() > 86400: continue
        
        # 简单指纹去重
        sig = n['title'][:15].lower()
        if sig in seen: continue
        seen.add(sig)
        candidates.append(n)
    
    final_news = ai_score_news(ai_key, candidates[:35])
    return price, kline, final_news

# =====================================================
# III. 技术处理 (修复核心)
# =====================================================

def process_kline_with_realtime(kline, current_price):
    """
    处理K线数据，并将实时价格注入到最后一根K线，消除数据源延迟带来的误差。
    """
    if not kline or len(kline) < 10: return pd.DataFrame()
    
    df = pd.DataFrame(kline)
    
    # 统一列名
    col_map = {
        't': 'Time', 'time': 'Time', 
        'o': 'Open', 'open': 'Open', 
        'h': 'High', 'high': 'High', 
        'l': 'Low', 'low': 'Low', 
        'c': 'Close','close':'Close'
    }
    df.rename(columns=col_map, inplace=True)
    
    try:
        # 1. 基础清洗
        for c in ['Open', 'High', 'Low', 'Close']: 
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df = df.dropna()
        if df.empty: return pd.DataFrame()

        # 2. 时间处理 (转换 + 时区调整)
        first_time = df['Time'].iloc[0]
        # 判断是毫秒还是秒
        unit = 'ms' if first_time > 10000000000 else 's'
        df['Time'] = pd.to_datetime(df['Time'], unit=unit)
        
        # 调整时区 (itick 默认为 UTC，这里+8转为北京时间用于显示，或者保持UTC)
        # 建议保持 API 原生时区逻辑，这里为了对其 +8
        df['Time'] = df['Time'] + timedelta(hours=8)

        # 3. 【核心修复】注入实时价格
        # 如果 FastForex 价格有效，强制更新最后一根K线的 Close
        # 这样图表最右侧和“现价”数字将完全一致
        if current_price > 0:
            last_idx = df.index[-1]
            
            # 更新收盘价
            df.at[last_idx, 'Close'] = current_price
            
            # 如果现价突破了当前K线的最高/最低，也需要更新，否则K线会很怪
            if current_price > df.at[last_idx, 'High']:
                df.at[last_idx, 'High'] = current_price
            if current_price < df.at[last_idx, 'Low']:
                df.at[last_idx, 'Low'] = current_price

        # 4. 重新计算指标 (必须在注入价格之后计算，否则 RSI 是错的)
        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta>0, 0)).fillna(0)
        loss = (-delta.where(delta<0, 0)).fillna(0)
        rs = gain.ewm(com=13).mean() / loss.ewm(com=13).mean()
        df['RSI'] = 100 - (100/(1+rs))
        
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean().fillna(0)
        
        return df.tail(100).reset_index(drop=True)
        
    except Exception as e:
        print(f"Data process error: {e}")
        return pd.DataFrame()

def plot_chart(df):
    if df.empty: return
    # 图表主题自适应：不强制 plotly_dark，让其背景透明，适应 Streamlit 的主题
    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.02)
    
    # K线 (颜色固定，因为红绿涨跌是金融通用的)
    fig.add_trace(go.Candlestick(
        x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], 
        name='Price',
        increasing_line_color='#00C805', decreasing_line_color='#FF4B4B'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['Time'], y=df['EMA20'], line=dict(color='#FFA500', width=1), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['EMA50'], line=dict(color='#00BFFF', width=1), name='EMA 50'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['Time'], y=df['RSI'], line=dict(color='#9400D3', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_color='red', line_dash='dot')
    fig.add_hline(y=30, row=2, col=1, line_color='green', line_dash='dot')
    
    # 关键：不设置 template='plotly_dark'，而是让背景透明
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', # 透明背景，适应浅色/深色模式
        plot_bgcolor='rgba(0,0,0,0)',
        height=500, 
        xaxis_rangeslider_visible=False, 
        dragmode='pan', 
        hovermode='x unified',
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(color=None) # 字体颜色自动
    )
    
    # 坐标轴颜色自适应设置比较复杂，这里给一个折中的灰色
    fig.update_xaxes(showgrid=False, color='gray')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', color='gray')
    
    config = {'scrollZoom': True, 'displayModeBar': True}
    st.plotly_chart(fig, use_container_width=True, config=config)

def generate_dual_report(api_key, market_data, news_list):
    if not api_key:
        return "⚠️ 请在侧边栏输入有效的 Gemini API Key 以启用分析功能。"

    client = genai.Client(api_key=api_key)
    ctx = "\n".join([f"- {n.get('title_zh', n['title'])} (Score: {n['ai_score']})" for n in news_list])
    prompt = f"""
    Role: Senior FX Strategist.
    [Technicals] Price:{market_data['price']} | RSI:{market_data['rsi']:.1f} | Trend:{market_data['trend']}
    [News] {ctx}
    [Output - CHINESE]
    ### 1. 核心决策
    **信号**: [🟢 BUY / 🔴 SELL / ⚖️ WAIT]
    **逻辑**: (Synthesize news + chart)
    ### 2. 详细分析
    * **基本面**: ...
    * **技术面**: ...
    """
    try:
        return client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text
    except: return "分析服务忙或 Key 无效。"

# =====================================================
# V. 主程序
# =====================================================

def main():
    inject_layout_css()
    
    # 默认初始化为空
    if 'key' not in st.session_state: st.session_state.key = YOUR_GEMINI_KEY

    with st.sidebar:
        st.title("🌗 Pro v42.1 (Custom)")
        st.caption("经典模式 + Key 配置")
        
        st.markdown("---")
        
        # 添加 API Key 输入框
        user_key = st.text_input("Gemini API Key", value=st.session_state.key, type="password", help="输入您的 Google Gemini API Key")
        st.session_state.key = user_key
        
        fmp_key = st.text_input("FMP Key", DEFAULT_FMP_KEY, type="password")
        ma_token = st.text_input("MarketAux", DEFAULT_MARKETAUX_TOKEN, type="password")
        period = st.selectbox("Timeframe", ["H1 (Hourly)", "D1 (Daily)"])
        k_type = 8 if "D1" in period else 2
        btn = st.button("🚀 刷新数据", type="primary")
        
        st.markdown("---")
        # 添加测试按钮
        if st.button("🧪 测试 Gemini Key"):
            if not user_key:
                st.error("请先输入 Key")
            elif not HAS_NEW_SDK:
                st.error("未检测到 SDK")
            else:
                try:
                    with st.spinner("验证连接中..."):
                        # 创建一个临时的 client 尝试发送请求
                        client = genai.Client(api_key=user_key)
                        resp = client.models.generate_content(model="gemini-2.5-flash", contents="Hi")
                        st.success(f"✅ Key 有效! 延迟: {resp.usage_metadata.total_token_count if resp.usage_metadata else 'N/A'}")
                except Exception as e:
                    st.error(f"❌ 连接失败: {str(e)}")

    st.title("EUR/USD Institutional Terminal")
    st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")

    if btn:
        with st.spinner("正在连接全球数据节点..."):
            # 获取数据
            realtime_price, kline, news = get_full_data(
                DEFAULT_FASTFOREX_API_KEY, DEFAULT_ITICK_TOKEN, fmp_key, ma_token, k_type, st.session_state.key
            )
            
            # 【关键】将实时价格传入处理函数
            df = process_kline_with_realtime(kline, realtime_price)
        
        if not df.empty:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 这里的 Close 已经被替换为 Realtime Price，所以 chg 是准确的
            chg = last['Close'] - prev['Close']
            trend = "多头" if last['EMA20'] > last['EMA50'] else "空头"
            
            # --- 1. OHLC 仪表盘 (原生组件，适应所有主题) ---
            st.subheader("📊 实时行情 (OHLC)")
            c1, c2, c3, c4, c5 = st.columns(5)
            
            # 颜色逻辑
            color_delta = "normal" # Streamlit 自动处理红绿
            
            c1.metric("现价 (Close)", f"{last['Close']:.5f}", f"{chg:.5f}")
            c2.metric("开盘 (Open)", f"{last['Open']:.5f}")
            c3.metric("最高 (High)", f"{last['High']:.5f}")
            c4.metric("最低 (Low)", f"{last['Low']:.5f}")
            c5.metric("RSI / 趋势", f"{last['RSI']:.1f}", trend)
            
            st.markdown("---")
            
            # --- 2. 主图表区 ---
            c_left, c_right = st.columns([2, 1])
            
            with c_left:
                st.subheader("📉 价格行为")
                plot_chart(df)
                
                # AI 报告 - 使用 st.info 自适应颜色
                if news:
                    st.subheader("🧠 首席策略分析")
                    with st.spinner("AI 思考中..."):
                        market_data = {"price": last['Close'], "trend": trend, "rsi": last['RSI']}
                        rpt = generate_dual_report(st.session_state.key, market_data, news)
                        # 使用原生容器，背景色会自动变
                        with st.container(border=True):
                            st.markdown(rpt)

            # --- 3. 新闻区 (右侧) ---
            with c_right:
                st.subheader("🦅 鹰眼情报")
                if not news:
                    st.info("过去 24 小时无高分新闻。")
                else:
                    for n in news:
                        score = n['ai_score']
                        if score >= 90: border_cls = "border-red"; badge = "🔥 核心"
                        elif score >= 80: border_cls = "border-orange"; badge = "⭐ 重要"
                        else: border_cls = "border-blue"; badge = "📝 资讯"
                        
                        src = "🇺🇸 Intl" if n['lang'] == 'en' else "🇨🇳 CN"
                        
                        # HTML 结构 + 注入的 CSS 类
                        html_card = f"""
                        <div class="news-box {border_cls}">
                            <div class="news-meta">{badge} ({score}) | {src} {n['source']}</div>
                            <div class="news-title">{n['title_zh']}</div>
                        </div>
                        """
                        st.markdown(html_card, unsafe_allow_html=True)
                        
                        if n['url'].startswith('http'):
                            st.link_button("🔗 原文", n['url'])

if __name__ == "__main__":
    main()
