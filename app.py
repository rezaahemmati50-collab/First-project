# app.py
"""
Crypto Signal Dashboard - Final integrated app.py
English UI. Includes:
- Header graphic & title
- Multi-select tickers + manual input
- Indicators: MA20/50/200, RSI14, MACD diff
- Buy/Hold/Sell signal (simple ensemble)
- Prophet forecasting (3/7/30 days) with safe handling
- Plotly interactive charts + CSV downloads
- News section from local data/sample_news.csv (fallback)
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
from datetime import datetime, date
import io
import os

# ----------------- Page config -----------------
st.set_page_config(page_title="Crypto Signal Dashboard", page_icon="🚀", layout="wide")
# CSS for header
st.markdown(
    """
    <style>
    .header {
        display:flex; align-items:center; gap:18px;
        padding:12px; border-radius:10px;
        background: linear-gradient(90deg,#0f172a,#0b1220);
        color: #f8fafc;
    }
    .logo {
        width:88px; height:88px; border-radius:12px; background:linear-gradient(135deg,#f6d365,#fda085);
        display:flex; align-items:center; justify-content:center; font-weight:800; color:#0b1220;
    }
    .title { font-size:26px; margin:0; font-weight:700; }
    .subtitle { font-size:13px; color:#e2e8f0; margin-top:6px; }
    </style>
    """, unsafe_allow_html=True
)

# header with logo placeholder (user can replace images/header.png)
col1, col2 = st.columns([1, 9])
with col1:
    try:
        st.markdown('<div class="logo">CF</div>', unsafe_allow_html=True)
    except Exception:
        st.write("CF")
with col2:
    st.markdown('<div class="header"><div><h1 class="title">Crypto Signal Dashboard</h1>'
                '<div class="subtitle">Live price, technical signals, and Prophet forecasting — Not financial advice</div>'
                '</div></div>', unsafe_allow_html=True)

st.write("---")

# ----------------- Helpers -----------------
@st.cache_data(ttl=180)
def yf_download_safe(ticker: str, period: str = "3mo", interval: str = "1d"):
    """Download via yfinance; return DataFrame (may be MultiIndex)."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None:
            return pd.DataFrame(), "yfinance returned None"
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"yfinance exception: {e}"

def extract_close(df, prefer_ticker=None):
    """Robust extraction of Close series from dataframe returned by yfinance."""
    if df is None or df.empty:
        return None, "dataframe empty"
    # MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # try level names
        try:
            # look for ('Close', ticker) pattern
            if 'Close' in df.columns.get_level_values(0):
                candidates = [c for c in df.columns if c[0] == 'Close']
                if prefer_ticker:
                    for c in candidates:
                        if prefer_ticker in str(c[1]):
                            s = df[c].copy()
                            s.index = pd.to_datetime(s.index)
                            s.name = 'Close'
                            return s.dropna(), ""
                s = df[candidates[0]].copy()
                s.index = pd.to_datetime(s.index)
                s.name = 'Close'
                return s.dropna(), ""
            if 'Close' in df.columns.get_level_values(1):
                candidates = [c for c in df.columns if c[1] == 'Close']
                s = df[candidates[0]].copy()
                s.index = pd.to_datetime(s.index)
                s.name = 'Close'
                return s.dropna(), ""
        except Exception as e:
            return None, f"multiindex extraction error: {e}"
        return None, "multiindex found but no 'Close' level"
    # normal columns
    if 'Close' in df.columns:
        s = df['Close'].copy()
        s.index = pd.to_datetime(s.index)
        s.name = 'Close'
        return s.dropna(), ""
    # try variants
    for cand in ['Adj Close','Adj_Close','adjclose','close']:
        if cand in df.columns:
            s = df[cand].copy()
            s.index = pd.to_datetime(s.index)
            s.name = 'Close'
            return s.dropna(), f"used {cand}"
    return None, "no close column found"

def safe_numeric_series(s):
    try:
        if isinstance(s, pd.Series):
            return pd.to_numeric(s, errors='coerce')
        return pd.to_numeric(pd.Series(s), errors='coerce')
    except Exception:
        return pd.Series(dtype='float64')

def compute_indicators_from_series(s: pd.Series):
    """Return DataFrame with Close, MA20, MA50, MA200, RSI14, MACD_diff."""
    s = safe_numeric_series(s).ffill().dropna()
    if s.empty:
        return pd.DataFrame()
    df = pd.DataFrame({'Close': s})
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(200, min_periods=1).mean()
    # RSI (14) using Wilder's smoothing approx with EMA
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    ma_down = down.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    df['RSI14'] = 100 - (100 / (1 + rs))
    # MACD diff
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = macd - signal
    return df

def generate_ensemble_signal(df_ind: pd.DataFrame):
    """Simple ensemble signal based on MA crossover, RSI, MACD"""
    if df_ind.empty:
        return "NO DATA"
    last = df_ind.iloc[-1]
    score = 0
    # MA rule
    try:
        if last['Close'] > last['MA20'] > last['MA50']:
            score += 1
        elif last['Close'] < last['MA20'] < last['MA50']:
            score -= 1
    except Exception:
        pass
    # RSI
    try:
        rsi = last['RSI14']
        if not np.isnan(rsi):
            if rsi < 30:
                score += 1
            elif rsi > 70:
                score -= 1
    except Exception:
        pass
    # MACD
    try:
        macd = last['MACD_diff']
        if not np.isnan(macd):
            score += 1 if macd > 0 else -1
    except Exception:
        pass
    # interpret
    if score >= 2:
        return "STRONG BUY"
    if score == 1:
        return "BUY"
    if score == 0:
        return "HOLD"
    if score == -1:
        return "SELL"
    return "STRONG SELL"

def prepare_prophet_df(series: pd.Series):
    df = series.dropna().to_frame(name='y').reset_index()
    df = df.rename(columns={df.columns[0]: 'ds', 'y': 'y'})
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['ds','y'])
    if df.shape[0] < 2:
        raise ValueError("Less than 2 valid rows for Prophet.")
    return df

def run_prophet_forecast(series: pd.Series, days: int = 3):
    try:
        df = prepare_prophet_df(series)
        m = Prophet(daily_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=days)
        forecast = m.predict(future)
        return forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(days), None
    except Exception as e:
        return None, str(e)

def plot_with_forecast(df_ind, forecast_df=None, title="Price"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['Close'], name='Close', line=dict(color='#1f77b4')))
    if 'MA20' in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MA20'], name='MA20', line=dict(color='#ff7f0e')))
    if 'MA50' in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MA50'], name='MA50', line=dict(color='#2ca02c')))
    if forecast_df is not None and not forecast_df.empty:
        # forecast
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast', line=dict(dash='dash', color='#d62728')))
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
            y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(214,39,40,0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=False
        ))
    fig.update_layout(title=title, xaxis_rangeslider_visible=True, height=420)
    return fig

def read_local_news(limit=10):
    sample = os.path.join("data","sample_news.csv")
    if not os.path.exists(sample):
        return []
    try:
        df = pd.read_csv(sample)
        out = []
        for _, r in df.head(limit).iterrows():
            out.append({"date": r.get("date"), "title": r.get("title"), "source": r.get("source"), "url": r.get("url")})
        return out
    except Exception:
        return []

# ----------------- Sidebar Inputs -----------------
st.sidebar.header("Inputs & Settings")
mode = st.sidebar.radio("Input mode", ["Manual tickers", "Upload CSV (ds,y)"])
if mode == "Manual tickers":
    default_list = "BTC-USD,ETH-USD,ADA-USD,SOL-USD,XRP-USD"
    chosen = st.sidebar.text_input("Tickers (comma separated)", value=default_list)
    tickers = [t.strip().upper() for t in chosen.split(",") if t.strip()]
else:
    uploaded = st.sidebar.file_uploader("Upload CSV (ds,y)", type=["csv"])

history = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)
forecast_days = st.sidebar.selectbox("Forecast (days)", [3,7,30], index=0)
enable_prophet = st.sidebar.checkbox("Enable Prophet", value=True)
show_news = st.sidebar.checkbox("Show News", value=True)
download_summary = st.sidebar.checkbox("Enable summary CSV download", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: For CSV upload use columns `ds` (date) and `y` (price).")

# ----------------- Main -----------------
st.header("Market Analysis")

if mode == "Upload CSV (ds,y)":
    if uploaded is None:
        st.info("Upload a CSV file with columns `ds` and `y` to analyze offline data.")
        st.stop()
    try:
        df_csv = pd.read_csv(uploaded)
        cols_lower = {c.lower(): c for c in df_csv.columns}
        if 'ds' in cols_lower and 'y' in cols_lower:
            ds_col = cols_lower['ds']; y_col = cols_lower['y']
            df_csv[ds_col] = pd.to_datetime(df_csv[ds_col], errors='coerce')
            df_csv[y_col] = pd.to_numeric(df_csv[y_col], errors='coerce')
            df_csv = df_csv.dropna(subset=[ds_col,y_col]).set_index(ds_col).sort_index()
            series = df_csv[y_col]
            df_ind = compute_indicators_from_series(series)
            st.subheader("Uploaded Data - Indicators")
            st.dataframe(df_ind.tail(8))
            st.markdown(f"**Signal:** {generate_ensemble_signal(df_ind)}")
            if enable_prophet:
                fc, ferr = run_prophet_forecast(series, days=forecast_days)
                if ferr:
                    st.warning(f"Forecast not available: {ferr}")
                else:
                    fc['ds'] = pd.to_datetime(fc['ds']).dt.date
                    st.subheader("Forecast (Prophet)")
                    st.table(fc.reset_index(drop=True))
                    st.download_button("Download forecast CSV", fc.to_csv(index=False).encode('utf-8'), file_name="uploaded_forecast.csv", mime="text/csv")
        else:
            st.error("Uploaded CSV must contain 'ds' and 'y' columns (case-insensitive).")
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")
else:
    # Manual tickers path - allow multi-select
    st.subheader("Select tickers to analyze")
    sel = st.multiselect("Tickers", options=tickers, default=tickers[:3])
    if not sel:
        st.info("Choose at least one ticker.")
        st.stop()

    combined_rows = []
    for t in sel:
        st.markdown(f"---\n### {t}")
        df_raw, err = yf_download_safe(t, period=history, interval=interval)
        if err:
            st.error(f"Error fetching {t}: {err}")
            continue
        close_s, dbg = extract_close(df_raw, prefer_ticker=t)
        if close_s is None or close_s.dropna().empty:
            st.error(f"Could not extract Close for {t}. debug: {dbg}")
            continue
        df_ind = compute_indicators_from_series(close_s)
        # show metrics
        last = df_ind.iloc[-1]
        try:
            last_close = float(last['Close'])
            st.metric(label=f"{t} Last Close (USD)", value=f"${last_close:,.6f}")
        except Exception:
            st.metric(label=f"{t} Last Close (USD)", value="N/A")
        signal = generate_ensemble_signal(df_ind)
        st.markdown(f"**Signal:** `{signal}`")
        st.dataframe(df_ind.tail(8).reset_index().rename(columns={df_ind.index.name or 'index': 'Date'}))
        # forecast
        forecast_df = None
        if enable_prophet:
            forecast_df, ferr = run_prophet_forecast(close_s, days=forecast_days)
            if ferr:
                st.warning(f"Prophet not available: {ferr}")
                forecast_df = None
        # plot
        fig = plot_with_forecast(df_ind, forecast_df, title=f"{t} Price & Forecast")
        st.plotly_chart(fig, use_container_width=True)
        # download forecast
        if forecast_df is not None:
            st.download_button(f"Download {t} forecast", forecast_df.to_csv(index=False).encode('utf-8'), file_name=f"{t}_forecast.csv", mime='text/csv')
        # append for combined CSV
        tail = df_ind.tail(5).reset_index().rename(columns={df_ind.index.name or 'index': 'Date'})
        tail['ticker'] = t
        combined_rows.append(tail[['ticker','Date','Close','MA20','MA50','RSI14']])

    # combined download
    if download_summary and combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
        st.download_button("Download combined last rows CSV", combined.to_csv(index=False).encode('utf-8'), file_name="summary_last_rows.csv", mime='text/csv')

# news
if show_news:
    st.markdown("---")
    st.header("News")
    news = read_local_news(limit=10)
    if not news:
        st.info("No local news found. Place data/sample_news.csv in the project folder to show sample news.")
    else:
        st.table(pd.DataFrame(news))

st.markdown("---")
st.caption("Made with ❤️ — Crypto Signal Dashboard. Not financial advice.")
