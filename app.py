# app.py
"""
Crypto Signal Dashboard - Single-file Final
Features:
- English UI with attractive header
- Multi-select tickers + manual input
- Upload CSV (ds,y) for offline forecasting
- Technical indicators: MA20/50/200, RSI14, MACD diff
- Ensemble buy/hold/sell signal
- Prophet forecasting (optional; auto-disabled if not installed)
- News via NewsAPI (optional) with local sample fallback
- Interactive Plotly charts and CSV downloads
- Defensive handling for yfinance MultiIndex outputs and missing data
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import io
import requests
import os
import textwrap

# Try to import Prophet; if not available, disable forecasting gracefully
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ---------------- Page config ----------------
st.set_page_config(page_title="Crypto Signal Dashboard", page_icon="🚀", layout="wide")
# CSS + Header
st.markdown(
    """
    <style>
    .header {
        display:flex; align-items:center; gap:18px;
        padding:14px; border-radius:12px;
        background: linear-gradient(90deg,#0f172a,#07112a);
        color: #f8fafc;
        margin-bottom: 12px;
    }
    .logo {
        width:88px; height:88px; border-radius:14px; 
        background: linear-gradient(135deg,#f6d365,#fda085);
        display:flex; align-items:center; justify-content:center; 
        font-weight:900; color:#07112a; font-size:28px;
    }
    .title { font-size:22px; margin:0; font-weight:700; }
    .subtitle { font-size:13px; color:#cbd5e1; margin-top:6px; }
    .small { font-size:12px; color:#94a3b8; }
    </style>
    """, unsafe_allow_html=True
)

col_a, col_b = st.columns([1, 8])
with col_a:
    st.markdown('<div class="logo">CF</div>', unsafe_allow_html=True)
with col_b:
    st.markdown(
        '<div class="header"><div><h1 class="title">Crypto Signal Dashboard</h1>'
        '<div class="subtitle">Live prices · Technical indicators · Forecasting (optional)</div>'
        '</div></div>',
        unsafe_allow_html=True
    )

st.write("")  # spacing

# ---------------- Utilities ----------------
@st.cache_data(ttl=120)
def fetch_yfinance(ticker: str, period: str = "3mo", interval: str = "1d"):
    """
    Download from yfinance and return DataFrame.
    Keeps defensive handling for MultiIndex columns.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None:
            return pd.DataFrame(), "yfinance returned None"
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"yfinance exception: {e}"

def extract_close_from_df(df: pd.DataFrame, prefer_ticker: str = None):
    """
    Return pd.Series of Close prices (tz-naive) from a yfinance df (normal or MultiIndex).
    """
    if df is None or df.empty:
        return None, "empty dataframe"
    # MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # try common patterns
            # pattern: ('Close', ticker) or (ticker, 'Close')
            if 'Close' in df.columns.get_level_values(0):
                close_cols = [c for c in df.columns if c[0] == 'Close']
                if prefer_ticker:
                    for c in close_cols:
                        if prefer_ticker in str(c[1]):
                            s = df[c].copy()
                            s.index = pd.to_datetime(s.index)
                            s = s.tz_convert(None) if getattr(s.index, 'tz', None) else s
                            s.name = 'Close'
                            return s.dropna(), ""
                s = df[close_cols[0]].copy()
                s.index = pd.to_datetime(s.index)
                s = s.tz_convert(None) if getattr(s.index, 'tz', None) else s
                s.name = 'Close'
                return s.dropna(), ""
            if 'Close' in df.columns.get_level_values(1):
                close_cols = [c for c in df.columns if c[1] == 'Close']
                s = df[close_cols[0]].copy()
                s.index = pd.to_datetime(s.index)
                s = s.tz_convert(None) if getattr(s.index, 'tz', None) else s
                s.name = 'Close'
                return s.dropna(), ""
        except Exception as e:
            return None, f"multiindex extraction error: {e}"
        return None, "MultiIndex but Close not found"
    # normal columns
    for cand in ['Close', 'close', 'Adj Close', 'Adj_Close', 'AdjClose']:
        if cand in df.columns:
            s = df[cand].copy()
            s.index = pd.to_datetime(s.index)
            try:
                if s.index.tz is not None:
                    s.index = s.index.tz_convert(None)
            except Exception:
                pass
            s.name = 'Close'
            return s.dropna(), f"used {cand}"
    return None, "No Close column found"

def to_numeric_safe(s):
    try:
        return pd.to_numeric(s, errors='coerce')
    except Exception:
        return pd.Series(dtype='float64')

# Indicators (no external 'ta' dependency)
def compute_indicators(series: pd.Series):
    """
    Input: series indexed by datetime
    Returns: DataFrame with Close, MA20, MA50, MA200, RSI14, MACD_diff
    """
    s = to_numeric_safe(series).ffill().dropna()
    if s.empty:
        return pd.DataFrame()
    df = pd.DataFrame({'Close': s})
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    # RSI (14) via EWMA smoothing (Wilder-like)
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    ma_down = down.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    df['RSI14'] = 100 - (100 / (1 + rs))
    # MACD diff (12-26, signal 9)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = macd - signal
    return df

def ensemble_signal(df_ind: pd.DataFrame):
    """
    Simple ensemble:
      score from MA crossover, RSI, MACD diff
    """
    if df_ind.empty:
        return "NO DATA"
    last = df_ind.iloc[-1]
    score = 0
    try:
        if last['Close'] > last['MA20'] > last['MA50']:
            score += 1
        elif last['Close'] < last['MA20'] < last['MA50']:
            score -= 1
    except Exception:
        pass
    rsi = last.get('RSI14', np.nan)
    if not np.isnan(rsi):
        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1
    macd = last.get('MACD_diff', np.nan)
    if not np.isnan(macd):
        score += 1 if macd > 0 else -1
    if score >= 2:
        return "STRONG BUY"
    elif score == 1:
        return "BUY"
    elif score == 0:
        return "HOLD"
    elif score == -1:
        return "SELL"
    else:
        return "STRONG SELL"

def run_prophet_forecast(series: pd.Series, days: int = 3):
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not available in the environment.")
    df = series.dropna().to_frame('y').reset_index().rename(columns={series.index.name or series.name: 'ds', 'y':'y'})
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['ds','y'])
    if df.shape[0] < 2:
        raise ValueError("Not enough data for Prophet (need at least 2 valid rows).")
    m = Prophet(daily_seasonality=True)
    m.fit(df[['ds','y']])
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(days)

# News utilities: try NewsAPI if key provided, else fallback to local sample
SAMPLE_NEWS = [
    {"date":"2025-08-01","title":"Market: BTC breaks resistance","source":"CryptoNews","url":"https://example.com/1"},
    {"date":"2025-07-30","title":"Ethereum upgrade announced","source":"CoinDaily","url":"https://example.com/2"},
    {"date":"2025-07-28","title":"Altcoins rally after news","source":"CoinBlog","url":"https://example.com/3"}
]

def fetch_news_from_newsapi(api_key: str, q: str = "crypto OR bitcoin OR ethereum", page_size: int = 10):
    try:
        url = "https://newsapi.org/v2/everything"
        params = {"q": q, "pageSize": page_size, "language":"en", "sortBy":"publishedAt", "apiKey": api_key}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        articles = []
        for a in data.get("articles", []):
            articles.append({"date": a.get("publishedAt"), "title": a.get("title"), "source": a.get("source",{}).get("name"), "url": a.get("url")})
        return articles
    except Exception:
        return []

# ---------------- Sidebar (inputs) ----------------
st.sidebar.header("Settings")
input_mode = st.sidebar.radio("Input mode", ["Manual tickers", "Upload CSV (ds,y)"])
if input_mode == "Manual tickers":
    presets = ["BTC-USD","ETH-USD","ADA-USD","SOL-USD","XRP-USD","DOGE-USD","LTC-USD","DOT-USD"]
    tickers_text = st.sidebar.text_input("Tickers (comma separated) — examples: BTC-USD,ETH-USD", value="BTC-USD,ETH-USD")
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
else:
    uploaded = st.sidebar.file_uploader("Upload CSV (ds,y)", type=["csv"])

history = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)
forecast_days = st.sidebar.selectbox("Forecast days", [3,7,30], index=0)
enable_prophet = st.sidebar.checkbox("Enable Prophet forecasting (requires 'prophet' package)", value=PROPHET_AVAILABLE)
if enable_prophet and not PROPHET_AVAILABLE:
    st.sidebar.warning("Prophet package not installed — forecasting disabled.")
show_news = st.sidebar.checkbox("Show news (NewsAPI or sample)", value=True)
newsapi_key = st.sidebar.text_input("NewsAPI key (optional)", value="")
download_combined = st.sidebar.checkbox("Enable combined CSV download", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: If Prophet installation fails in your environment, uncheck Prophet to keep the app working.")

# ---------------- Main ----------------
st.header("Market Analysis & Signals")

# CSV upload path
if input_mode == "Upload CSV (ds,y)":
    if 'uploaded' not in locals():
        st.info("Please upload a CSV file with columns: ds (date), y (price).")
        st.stop()
    if uploaded is None:
        st.info("Upload a CSV file to proceed.")
        st.stop()
    # parse CSV
    try:
        df_csv = pd.read_csv(uploaded)
        cols_lower = {c.lower(): c for c in df_csv.columns}
        if 'ds' in cols_lower and 'y' in cols_lower:
            ds_col = cols_lower['ds']; y_col = cols_lower['y']
            df_csv[ds_col] = pd.to_datetime(df_csv[ds_col], errors='coerce')
            df_csv[y_col] = pd.to_numeric(df_csv[y_col], errors='coerce')
            df_csv = df_csv.dropna(subset=[ds_col,y_col]).set_index(ds_col).sort_index()
            series = df_csv[y_col]
            df_ind = compute_indicators(series)
            st.subheader("Uploaded Data Indicators")
            st.dataframe(df_ind.tail(10))
            st.markdown(f"**Signal:** {ensemble_signal(df_ind)}")
            if enable_prophet and PROPHET_AVAILABLE:
                try:
                    fc = run_prophet_forecast(series, days=forecast_days)
                    fc['ds'] = pd.to_datetime(fc['ds']).dt.date
                    st.subheader(f"Forecast ({forecast_days} days)")
                    st.table(fc.reset_index(drop=True))
                    st.download_button("Download forecast CSV", fc.to_csv(index=False).encode('utf-8'), file_name="uploaded_forecast.csv", mime="text/csv")
                except Exception as e:
                    st.warning(f"Forecast failed: {e}")
        else:
            st.error("Uploaded CSV must include columns 'ds' and 'y' (case-insensitive).")
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")
else:
    # Manual tickers: allow multi-select
    st.subheader("Choose tickers")
    sel = st.multiselect("Select tickers to analyze", options=tickers, default=tickers[:2])
    if not sel:
        st.info("Select at least one ticker.")
        st.stop()

    combined_rows = []
    for t in sel:
        st.markdown("---")
        st.subheader(f"{t}")
        df_raw, err = fetch_yfinance(t, period=history, interval=interval)
        if err:
            st.error(f"Error fetching {t}: {err}")
            continue
        close_series, dbg = extract_close_from_df(df_raw, prefer_ticker=t)
        if close_series is None or close_series.dropna().empty:
            st.error(f"Could not extract Close for {t}. Debug: {dbg}")
            continue
        df_ind = compute_indicators(close_series)
        if df_ind.empty:
            st.error(f"No valid indicator data for {t}.")
            continue
        # metrics
        last = df_ind.iloc[-1]
        try:
            last_price = float(last['Close'])
            st.metric(label=f"{t} Latest Close (USD)", value=f"${last_price:,.6f}")
        except Exception:
            st.metric(label=f"{t} Latest Close (USD)", value="N/A")
        sig = ensemble_signal(df_ind)
        st.markdown(f"**Signal:** `{sig}`")
        # show table summary
        st.dataframe(df_ind.tail(8).reset_index().rename(columns={df_ind.index.name or 'index': 'Date'}))
        # forecast if enabled
        forecast_df = None
        if enable_prophet and PROPHET_AVAILABLE:
            try:
                forecast_df, ferr = (run_prophet_forecast(close_series, days=forecast_days), None)
            except Exception as e:
                forecast_df, ferr = None, str(e)
            if ferr:
                st.warning(f"Forecast unavailable: {ferr}")
            else:
                # display forecast summary table
                fc = forecast_df.copy()
                fc['ds'] = pd.to_datetime(fc['ds']).dt.date
                st.subheader(f"{t} Forecast ({forecast_days} days)")
                st.table(fc.reset_index(drop=True))
                st.download_button(f"Download {t} forecast", forecast_df.to_csv(index=False).encode('utf-8'), file_name=f"{t}_forecast.csv", mime="text/csv")
        elif enable_prophet and not PROPHET_AVAILABLE:
            st.info("Prophet not installed. Uncheck Prophet in sidebar or install the package to enable forecasts.")
        # interactive plots: price+MAs+forecast; RSI and MACD as small charts
        try:
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(x=df_ind.index, y=df_ind['Close'], name='Close', line=dict(color='#1f77b4')))
            if 'MA20' in df_ind.columns:
                fig_main.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MA20'], name='MA20', line=dict(color='#ff7f0e')))
            if 'MA50' in df_ind.columns:
                fig_main.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MA50'], name='MA50', line=dict(color='#2ca02c')))
            if forecast_df is not None:
                fig_main.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast', line=dict(dash='dash', color='#d62728')))
                fig_main.add_trace(go.Scatter(
                    x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
                    y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]),
                    fill='toself', fillcolor='rgba(214,39,40,0.08)', line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=False))
            fig_main.update_layout(title=f"{t} Price", xaxis_rangeslider_visible=True, height=460)
            st.plotly_chart(fig_main, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not render main chart: {e}")
        # RSI chart
        try:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_ind.index, y=df_ind['RSI14'], name='RSI14', line=dict(color='#9467bd')))
            fig_rsi.update_layout(title=f"{t} RSI (14)", yaxis=dict(range=[0,100]), height=250)
            st.plotly_chart(fig_rsi, use_container_width=True)
        except Exception:
            pass
        # MACD chart
        try:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=df_ind.index, y=df_ind['MACD_diff'], name='MACD diff'))
            fig_macd.update_layout(title=f"{t} MACD diff", height=250)
            st.plotly_chart(fig_macd, use_container_width=True)
        except Exception:
            pass
        # prepare combined rows for CSV
        tail = df_ind.tail(5).reset_index().rename(columns={df_ind.index.name or 'index':'Date'})
        tail['ticker'] = t
        combined_rows = tail[['ticker','Date','Close','MA20','MA50','RSI14','MACD_diff']].copy()
        combined_rows['Date'] = combined_rows['Date'].astype(str)
        if 'all_combined' not in st.session_state:
            st.session_state.all_combined = []
        st.session_state.all_combined.append(combined_rows)

    # combined download
    if download_combined and st.session_state.get('all_combined'):
        try:
            combined_df = pd.concat(st.session_state.all_combined, ignore_index=True)
            st.download_button("Download combined summary CSV", combined_df.to_csv(index=False).encode('utf-8'), file_name='combined_summary.csv', mime='text/csv')
        except Exception:
            pass

# News section
if show_news:
    st.markdown("---")
    st.header("News")
    news_items = []
    if newsapi_key:
        news_items = fetch_news_from_newsapi(newsapi_key, q="crypto OR bitcoin OR ethereum", page_size=10)
    if not news_items:
        news_items = SAMPLE_NEWS
    if news_items:
        df_news = pd.DataFrame(news_items)
        st.dataframe(df_news)
    else:
        st.info("No news available.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ — Crypto Signal Dashboard. Not financial advice.")
