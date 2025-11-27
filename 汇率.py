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

# ⚠️ 原有的公共 Key 已因泄露被 Google 封禁。
# 请在侧边栏输入您自己的 Key。
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
    page_title="EUR/USD Pro v47.0 (Stable)",
    page_icon="🌗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# I. 极简 CSS
# =====================================================

def inject_layout_css():
    st.markdown("""
    <style>
        .news-box {
            border-left-width: 5px;
            border-left-style: solid;
            padding: 10px 15px;
            margin-bottom: 10px;
            background-color: rgba(128, 128, 128, 0.1); 
            border-radius: 0 5px 5px 0;
        }
        .border-red { border-left-color: #FF4B4B; }
        .border-orange { border-left-color: #FFA500; }
        .border-blue { border-left-color: #00BFFF; }
        .border-gray { border-left-color: #808080; }
        .news-meta { font-size: 0.85rem; opacity: 0.8; margin-bottom: 4px; }
        .news-title { font-size: 1.1rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

# =====================================================
# II. 数据抓取逻辑 (增强容错)
# =====================================================

class NewsFetcher:
    @staticmethod
    def fetch_google_cn():
        items = []
        # 尝试中文源
        try:
            query = "欧元+美元+汇率+OR+美联储+OR+欧洲央行"
            url = f"https://news.google.com/rss/search?q={query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans"
            headers = {'User-Agent': 'Mozilla/5.0'}
            r = requests.get(url, headers=headers, timeout=4)
            if r.status_code == 200:
                root = ET.fromstring(r.content)
                for item in root.findall('./channel/item'):
                    dt = date_parser.parse(item.find('pubDate').text)
                    items.append({"title": item.find('title').text, "url": item.find('link').text, "source": "Google(CN)", "lang": "zh", "dt": dt})
        except: pass
        
        # 如果中文源没数据，尝试英文源作为备用
        if not items:
            try:
                query_en = "EURUSD+Forex+ECB+Fed"
                url_en = f"https://news.google.com/rss/search?q={query_en}&hl=en-US&gl=US&ceid=US:en"
                r = requests.get(url_en, headers={'User-Agent': 'Mozilla/5.0'}, timeout=4)
                if r.status_code == 200:
                    root = ET.fromstring(r.content)
                    for item in root.findall('./channel/item'):
                        dt = date_parser.parse(item.find('pubDate').text)
                        items.append({"title": item.find('title').text, "url": item.find('link').text, "source": "Google(Intl)", "lang": "en", "dt": dt})
            except: pass
            
        return items

    @staticmethod
    def fetch_fmp(key):
        items = []
        if not key: return []
        try:
            url = f"https://financialmodelingprep.com/api/v4/forex_news?symbol=EURUSD&limit=20&apikey={key}"
            r = requests.get(url, timeout=5)
            data = r.json()
            if isinstance(data, list):
                for i in data:
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
            params = {"api_token": token, "search": 'EURUSD OR "Fed" OR "ECB"', "language": "en", "limit": 20}
            r = requests.get(url, params=params, timeout=6)
            if 'data' in r.json():
                for i in r.json()['data']:
                    dt = date_parser.parse(i['published_at'])
                    items.append({"title": i['title'], "url": i['url'], "source": "MarketAux", "lang": "en", "dt": dt})
        except: pass
        return items

def ai_score_news(api_key, raw_news_list):
    """
    AI 评分。如果 Key 无效，返回 raw 数据。
    """
    if not raw_news_list: return []
    
    # 准备保底数据
    fallback_news = []
    for item in raw_news_list[:10]:
        item['title_zh'] = item['title']
        item['ai_score'] = 0
        item['status'] = 'raw' 
        fallback_news.append(item)

    # 如果没有输入 Key，直接返回保底
    if not api_key or not HAS_NEW_SDK: 
        return fallback_news

    client = genai.Client(api_key=api_key)
    process_list = raw_news_list[:15]
    payload = "".join([f"ID {i}: {n['title']}\n" for i, n in enumerate(process_list)])
    
    prompt = f"""
    You are a FX News Filter.
    Task: Translate titles to Chinese (if needed) and Score Relevance (0-100) for EUR/USD.
    Rules: 
    - 80-100: Central Banks, NFP, CPI, War.
    - 50-79: Econ Data, Markets.
    - 0-49: Noise.
    Input:\n{payload}
    Output JSON ONLY: [ {{"id": 0, "score": 85, "zh_title": "..."}}, ... ]
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
                if idx < len(process_list):
                    item = process_list[idx]
                    item['title_zh'] = s.get('zh_title', item['title'])
                    item['ai_score'] = score
                    item['status'] = 'scored'
                    if score >= 50: 
                        high_value_items.append(item)
            
            high_value_items.sort(key=lambda x: x['ai_score'], reverse=True)
            if not high_value_items: return fallback_news 
            return high_value_items
        
        return fallback_news
    except Exception:
        return fallback_news

def get_full_data(fast_key, itick_token, fmp_key, ma_token, k_type, ai_key):
    price = 0
    kline = []
    
    with ThreadPoolExecutor(max_workers=5) as ex:
        t_price = ex.submit(requests.get, f"https://api.fastforex.io/fetch-one?from=EUR&to=USD&api_key={fast_key}")
        t_kline = ex.submit(requests.get, f"https://api.itick.org/forex/kline?region=GB&code={SYMBOL}&kType={k_type}&limit=200", headers={"token": itick_token})
        t_cn = ex.submit(NewsFetcher.fetch_google_cn)
        t_fmp = ex.submit(NewsFetcher.fetch_fmp, fmp_key)
        t_ma = ex.submit(NewsFetcher.fetch_marketaux, ma_token)
        
        try: 
            price_res = t_price.result().json()
            if 'result' in price_res and 'EURUSD' in price_res['result']:
                price = float(price_res['result']['EURUSD'])
        except: price = 0

        try: 
            r = t_kline.result().json()
            if r.get('code') == 0: kline = r.get('data', [])
        except: pass
        
        try: raw_news = t_cn.result() + t_fmp.result() + t_ma.result()
        except: raw_news = []

    now = datetime.now(timezone.utc)
    
    # --- 智能筛选逻辑 ---
    raw_news.sort(key=lambda x: x['dt'], reverse=True)
    
    # 1. 尝试只取24小时内的
    valid_news = []
    for n in raw_news:
        if n['dt'].tzinfo is None: n['dt'] = n['dt'].replace(tzinfo=timezone.utc)
        else: n['dt'] = n['dt'].astimezone(timezone.utc)
        
        if (now - n['dt']).total_seconds() <= 86400:
            valid_news.append(n)
            
    # 2. 如果有效新闻太少(<3)，触发保底机制，直接取最新的5条（不管时间）
    # 这能解决“AI用不了，新闻也看不见”的问题
    if len(valid_news) < 3:
        candidates = raw_news[:5] 
    else:
        candidates = valid_news

    # 去重
    unique_candidates = []
    seen = set()
    for n in candidates:
         sig = n['title'][:15].lower()
         if sig in seen: continue
         seen.add(sig)
         unique_candidates.append(n)
    
    final_news = ai_score_news(ai_key, unique_candidates[:35])
    return price, kline, final_news, len(raw_news)

# =====================================================
# III. 技术处理
# =====================================================

def process_kline_with_realtime(kline, current_price):
    if not kline or len(kline) < 10: return pd.DataFrame()
    df = pd.DataFrame(kline)
    col_map = {'t': 'Time', 'time': 'Time', 'o': 'Open', 'open': 'Open', 'h': 'High', 'high': 'High', 'l': 'Low', 'low': 'Low', 'c': 'Close','close':'Close'}
    df.rename(columns=col_map, inplace=True)
    try:
        for c in ['Open', 'High', 'Low', 'Close']: df[c] = pd.to_numeric(df[c], errors='coerce')
        df['Time'] = pd.to_numeric(df['Time'], errors='coerce')
        df = df.dropna()
        if df.empty: return pd.DataFrame()
        first_time = df['Time'].iloc[0]
        unit = 'ms' if first_time > 10000000000 else 's'
        df['Time'] = pd.to_datetime(df['Time'], unit=unit) + timedelta(hours=8)

        if current_price > 0:
            last_idx = df.index[-1]
            df.at[last_idx, 'Close'] = current_price
            if current_price > df.at[last_idx, 'High']: df.at[last_idx, 'High'] = current_price
            if current_price < df.at[last_idx, 'Low']: df.at[last_idx, 'Low'] = current_price

        df['EMA20'] = df['Close'].ewm(span=20).mean()
        df['EMA50'] = df['Close'].ewm(span=50).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta>0, 0)).fillna(0)
        loss = (-delta.where(delta<0, 0)).fillna(0)
        rs = gain.ewm(com=13).mean() / loss.ewm(com=13).mean()
        df['RSI'] = 100 - (100/(1+rs))
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean().fillna(0)
        return df.tail(100).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def plot_chart(df):
    if df.empty: return
    fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df['Time'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price', increasing_line_color='#00C805', decreasing_line_color='#FF4B4B'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['EMA20'], line=dict(color='#FFA500', width=1), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['EMA50'], line=dict(color='#00BFFF', width=1), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Time'], y=df['RSI'], line=dict(color='#9400D3', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, row=2, col=1, line_color='red', line_dash='dot')
    fig.add_hline(y=30, row=2, col=1, line_color='green', line_dash='dot')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=500, xaxis_rangeslider_visible=False, dragmode='pan', hovermode='x unified', margin=dict(l=10, r=10, t=10, b=10), font=dict(color=None))
    fig.update_xaxes(showgrid=False, color='gray')
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.2)', color='gray')
    config = {'scrollZoom': True, 'displayModeBar': True}
    st.plotly_chart(fig, use_container_width=True, config=config)

def generate_dual_report(api_key, market_data, news_list):
    # 检查 Key 是否为空
    if not api_key:
        return "⚠️ 未检测到有效 Key。AI 深度分析已暂停。但上方技术图表和右侧原始新闻依然可用。"

    is_raw_mode = any(n.get('status') == 'raw' for n in news_list)
    
    if not news_list:
        ctx = "No news data available."
    elif is_raw_mode:
        ctx = "Note: AI News Scoring Skipped (Raw Mode). Headlines:\n" + "\n".join([f"- {n['title']}" for n in news_list[:5]])
    else:
        ctx = "\n".join([f"- {n.get('title_zh', n['title'])} (Score: {n.get('ai_score',0)})" for n in news_list])
        
    prompt = f"""
    Role: Senior FX Strategist.
    [Technicals] Price:{market_data['price']} | RSI:{market_data['rsi']:.1f} | Trend:{market_data['trend']}
    [Context] {ctx}
    
    [Output - CHINESE]
    ### 1. 核心决策
    **信号**: [🟢 BUY / 🔴 SELL / ⚖️ WAIT]
    **简述**: ...
    
    ### 2. 分析
    * **技术面**: ...
    * **消息面**: ...
    """
    
    client = genai.Client(api_key=api_key)
    try:
        return client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text
    except Exception as e:
        return f"⚠️ AI 分析失败 (请检查Key是否有效): {str(e)}"

# =====================================================
# V. 主程序
# =====================================================

def main():
    inject_layout_css()
    
    # 默认初始化为空，强迫用户输入
    if 'key' not in st.session_state: 
        st.session_state.key = YOUR_GEMINI_KEY

    with st.sidebar:
        st.title("🌗 Pro v47.0 (Stable)")
        st.caption("增加新闻强制保底显示")
        st.markdown("---")
        
        # 密码框
        user_key = st.text_input("Gemini API Key", value=st.session_state.key, type="password", help="如需AI分析和翻译，请填入Key")
        st.session_state.key = user_key
        
        if not user_key:
            st.info("ℹ️ 未输入 Key：将仅显示原始新闻和技术图表。")
            
        fmp_key = st.text_input("FMP Key", DEFAULT_FMP_KEY, type="password")
        ma_token = st.text_input("MarketAux", DEFAULT_MARKETAUX_TOKEN, type="password")
        period = st.selectbox("Timeframe", ["H1 (Hourly)", "D1 (Daily)"])
        k_type = 8 if "D1" in period else 2
        btn = st.button("🚀 刷新数据", type="primary")
        
        st.markdown("---")
        if st.button("🧪 测试 Key"):
            if not user_key:
                st.error("请先输入 Key")
            elif not HAS_NEW_SDK:
                st.error("未检测到 SDK")
            else:
                try:
                    with st.spinner("验证中..."):
                        client = genai.Client(api_key=user_key)
                        resp = client.models.generate_content(model="gemini-2.5-flash", contents="Hi")
                        st.success("✅ Key 有效！")
                except Exception as e:
                    st.error(f"❌ Key 无效:\n{str(e)}")

    st.title("EUR/USD Institutional Terminal")
    st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")

    if btn:
        with st.spinner("正在连接全球数据节点..."):
            realtime_price, kline, news, raw_count = get_full_data(
                DEFAULT_FASTFOREX_API_KEY, DEFAULT_ITICK_TOKEN, fmp_key, ma_token, k_type, st.session_state.key
            )
            df = process_kline_with_realtime(kline, realtime_price)
        
        if not df.empty:
            last = df.iloc[-1]
            prev = df.iloc[-2]
            chg = last['Close'] - prev['Close']
            trend = "多头" if last['EMA20'] > last['EMA50'] else "空头"
            
            st.subheader("📊 实时行情 (OHLC)")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("现价 (Close)", f"{last['Close']:.5f}", f"{chg:.5f}")
            c2.metric("开盘 (Open)", f"{last['Open']:.5f}")
            c3.metric("最高 (High)", f"{last['High']:.5f}")
            c4.metric("最低 (Low)", f"{last['Low']:.5f}")
            c5.metric("RSI / 趋势", f"{last['RSI']:.1f}", trend)
            st.markdown("---")
            
            c_left, c_right = st.columns([2, 1])
            with c_left:
                st.subheader("📉 价格行为")
                plot_chart(df)
                st.subheader("🧠 首席策略分析")
                with st.spinner("AI 正在分析..."):
                    market_data = {"price": last['Close'], "trend": trend, "rsi": last['RSI']}
                    rpt = generate_dual_report(st.session_state.key, market_data, news)
                    
                    if "⚠️" in rpt:
                        st.warning(rpt) 
                    else:
                        with st.container(border=True):
                            st.markdown(rpt)

            with c_right:
                st.subheader(f"🦅 鹰眼情报 (源: {raw_count})")
                if not news:
                    st.warning("暂无数据 (请检查网络)")
                else:
                    for n in news:
                        is_raw = n.get('status') == 'raw'
                        score = n.get('ai_score', 0)
                        
                        # 计算时间差显示
                        now_utc = datetime.now(timezone.utc)
                        diff_hr = (now_utc - n['dt']).total_seconds() / 3600
                        if diff_hr < 1: time_str = "刚刚"
                        elif diff_hr < 24: time_str = f"{int(diff_hr)}小时前"
                        else: time_str = f"{int(diff_hr/24)}天前"

                        if is_raw:
                            border_cls = "border-gray"
                            badge = "⚠️ AI未连接"
                        elif score >= 80: border_cls = "border-red"; badge = "🔥 核心"
                        elif score >= 50: border_cls = "border-orange"; badge = "⭐ 关注"
                        else: border_cls = "border-blue"; badge = "📰 资讯"
                        
                        src = "🇺🇸 Intl" if n['lang'] == 'en' else "🇨🇳 CN"
                        html_card = f"""
                        <div class="news-box {border_cls}">
                            <div class="news-meta">{badge} ({score}) | {src} {n['source']} | 🕒 {time_str}</div>
                            <div class="news-title">{n.get('title_zh', n['title'])}</div>
                        </div>
                        """
                        st.markdown(html_card, unsafe_allow_html=True)
                        if n['url'].startswith('http'):
                            st.link_button("🔗 原文", n['url'])

if __name__ == "__main__":
    main()
