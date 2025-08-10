# app.py -- CryptoForecast Final Integrated
# English UI. Ready for Streamlit Cloud / local.
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import datetime, date
import os
from news.news_utils import fetch_news  # helper that falls back to local CSV if no API key
import io

# ----------------- Page config -----------------
st.set_page_config(page_title="CryptoForecast — Final", layout="wide", page_icon="📈")
st.title("📈 CryptoForecast — Integrated Dashboard")
st.markdown("Live prices (Yahoo Finance), technical indicators, and Prophet forecasting. Not financial advice.")

# ----------------- Helpers -----------------
def safe_to_numeric(s):
    """Convert input to pd.Series numeric safely."""
    try:
        if isinstance(s, pd.Series):
            return pd.to_numeric(s, errors='coerce')
        return pd.to_numeric(pd.Series(s), errors='coerce')
    except Exception:
        # if can't convert, return empty numeric series
        try:
            return pd.Series([float(x) for x in list(s)])
        except Exception:
            return pd.Series(dtype='float64')

def extract_close_series(df, prefer_ticker=None):
    """
    Robustly extract 1-d Close series from yfinance DataFrame.
    Returns (pd.Series or None, message)
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None, "Data frame empty"
    # MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # try to find level named 'Close' at any level
        try:
            if 'Close' in df.columns.get_level_values(0):
                close_cols = [c for c in df.columns if c[0] == 'Close']
                # prefer ticker match if provided
                if prefer_ticker:
                    for c in close_cols:
                        if prefer_ticker in str(c[1]):
                            s = df[c].copy()
                            s.index = pd.to_datetime(s.index)
                            s.name = 'Close'
                            return s, f"Close extracted for {prefer_ticker} from multiindex (level 0)."
                # fallback first
                s = df[close_cols[0]].copy()
                s.index = pd.to_datetime(s.index)
                s.name = 'Close'
                return s, "Close extracted from multiindex level 0 (first column)."
            if 'Close' in df.columns.get_level_values(1):
                close_cols = [c for c in df.columns if c[1] == 'Close']
                if prefer_ticker:
                    for c in close_cols:
                        if prefer_ticker in str(c[0]):
                            s = df[c].copy()
                            s.index = pd.to_datetime(s.index)
                            s.name = 'Close'
                            return s, f"Close extracted for {prefer_ticker} from multiindex (level 1)."
                s = df[close_cols[0]].copy()
                s.index = pd.to_datetime(s.index)
                s.name = 'Close'
                return s, "Close extracted from multiindex level 1 (first column)."
        except Exception as e:
            return None, f"Error extracting Close from MultiIndex: {e}"
        return None, "MultiIndex found but 'Close' level not present."
    # normal columns
    if 'Close' in df.columns:
        s = df['Close'].copy()
        s.index = pd.to_datetime(df.index)
        s.name = 'Close'
        return s, "Close extracted from df['Close']."
    # try other likely names
    for candidate in ['Adj Close', 'Adj_Close', 'AdjClose', 'close', 'adjclose']:
        if candidate in df.columns:
            s = df[candidate].copy()
            s.index = pd.to_datetime(df.index)
            s.name = 'Close'
            return s, f"Using '{candidate}' as Close."
    return None, "No Close or Adj Close column found."

@st.cache_data(ttl=300)
def download_yf(ticker, period="3mo", interval="1d"):
    """Download via yfinance and return raw df (may be multiindex)."""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        return df
    except Exception as e:
        return pd.DataFrame()

def compute_indicators(series: pd.Series):
    """Return DataFrame with Close and indicators: MA20, MA50, MA200, RSI14, MACD_diff"""
    s = safe_to_numeric(series).ffill().dropna()
    if s.empty:
        return pd.DataFrame()
    df = pd.DataFrame({'Close': s})
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    # RSI simple implementation
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False, min_periods=14).mean()
    ema_down = down.ewm(com=13, adjust=False, min_periods=14).mean()
    rs = ema_up / (ema_down.replace(0, np.nan))
    df['RSI14'] = 100 - (100 / (1 + rs))
    # MACD diff (12-26 EMA)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = macd - signal
    return df

def generate_signal(df_ind):
    """Return signal string based on last row of indicators."""
    if df_ind.empty:
        return "NO DATA"
    last = df_ind.iloc[-1]
    score = 0
    # MA crossover
    if last['Close'] > last['MA20'] > last['MA50']:
        score += 1
    elif last['Close'] < last['MA20'] < last['MA50']:
        score -= 1
    # RSI
    rsi = last.get('RSI14', np.nan)
    if not np.isnan(rsi):
        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1
    # MACD
    macd = last.get('MACD_diff', np.nan)
    if not np.isnan(macd):
        score += 1 if macd > 0 else -1
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
    """Return DataFrame with ds,y for Prophet or raise Exception."""
    df = series.dropna().to_frame('y').reset_index()
    df = df.rename(columns={df.columns[0]: 'ds', 'y': 'y'})
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['ds', 'y'])
    if df.shape[0] < 2:
        raise ValueError("Not enough non-NaN rows for Prophet (need >=2).")
    return df

def run_prophet(series: pd.Series, periods=3):
    """Fit prophet and return forecast df (ds,yhat,yhat_lower,yhat_upper)."""
    df = prepare_prophet_df(series)
    m = Prophet(daily_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

def plot_series_with_indicators(df_ind, forecast_df=None, title="Price"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['Close'], name='Close'))
    if 'MA20' in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MA20'], name='MA20'))
    if 'MA50' in df_ind.columns:
        fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MA50'], name='MA50'))
    if forecast_df is not None and not forecast_df.empty:
        # plot forecast yhat
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast', line=dict(dash='dash')))
        # shaded area
        fig.add_trace(go.Scatter(
            x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
            y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(0,176,246,0.1)', line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=False
        ))
    fig.update_layout(title=title, xaxis_rangeslider_visible=True, height=450)
    return fig

# ----------------- Sidebar Inputs -----------------
st.sidebar.header("Settings")
input_mode = st.sidebar.radio("Input mode", ["Manual tickers", "Upload CSV (ds,y)"])
if input_mode == "Manual tickers":
    tickers_input = st.sidebar.text_input("Tickers (comma separated). Example: BTC-USD,ETH-USD", value="BTC-USD,ETH-USD")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file (ds,y)", type=['csv'])
    tickers = None

history = st.sidebar.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "1wk"], index=0)
forecast_choice = st.sidebar.selectbox("Forecast horizon (days)", [3, 7, 30], index=0)
enable_prophet = st.sidebar.checkbox("Enable Prophet forecasting", value=True)
show_news = st.sidebar.checkbox("Show News (sample/local or via NewsAPI)", value=True)
download_summary = st.sidebar.checkbox("Offer combined summary CSV", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: Upload CSV should have `ds` (date) and `y` (price) columns for offline forecasting.")

# ----------------- Main -----------------
# header graphic
colh1, colh2 = st.columns([1, 6])
with colh1:
    try:
        st.image("images/header.png", width=120)
    except Exception:
        st.write("📈")
with colh2:
    st.markdown("### CryptoForecast — Clean, fast & robust\n**Prices from Yahoo Finance, Prophet forecasting, technical indicators.**")

st.markdown("---")

results = {}  # store per ticker results for combined summary

if input_mode == "Upload CSV (ds,y)":
    if uploaded_file is None:
        st.info("Please upload a CSV file with columns `ds` and `y`.")
        st.stop()
    try:
        df_user = pd.read_csv(uploaded_file)
        if {'ds', 'y'}.issubset(set([c.lower() for c in df_user.columns])):
            # normalize column names
            cols_lower = {c.lower(): c for c in df_user.columns}
            ds_col = cols_lower['ds']
            y_col = cols_lower['y']
            df_user[ds_col] = pd.to_datetime(df_user[ds_col], errors='coerce')
            df_user[y_col] = pd.to_numeric(df_user[y_col], errors='coerce')
            df_user = df_user.dropna(subset=[ds_col, y_col]).set_index(ds_col).sort_index()
            series = df_user[y_col]
            df_ind = compute_indicators(series)
            st.subheader("Uploaded data indicators")
            st.dataframe(df_ind.tail(10))
            signal = generate_signal(df_ind)
            st.markdown(f"**Signal:** {signal}")
            if enable_prophet:
                try:
                    fc = run_prophet(series, periods=forecast_choice)
                    st.subheader("Forecast (Prophet)")
                    fc_display = fc.copy()
                    fc_display['ds'] = pd.to_datetime(fc_display['ds']).dt.date
                    st.table(fc_display.reset_index(drop=True))
                    csv_bytes = fc.to_csv(index=False).encode('utf-8')
                    st.download_button("Download forecast CSV", csv_bytes, file_name="uploaded_forecast.csv", mime='text/csv')
                except Exception as e:
                    st.error(f"Prophet forecast failed: {e}")
        else:
            st.error("Uploaded CSV missing 'ds' and 'y' columns (case-insensitive).")
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")
else:
    # manual tickers path
    st.subheader("Ticker analysis")
    for t in tickers:
        st.markdown(f"---\n### {t}")
        df_raw = download_yf(t, period=history, interval=interval)
        if df_raw is None or df_raw.empty:
            st.error(f"No data for {t}. Skipping.")
            continue
        close_ser, msg = extract_close_series(df_raw, prefer_ticker=t)
        st.write(f"*debug:* {msg}")
        if close_ser is None or close_ser.dropna().empty:
            st.error(f"Could not extract Close for {t}.")
            continue
        # build indicators df
        df_ind = compute_indicators(close_ser)
        results[t] = df_ind  # store
        # show key metrics
        last_row = df_ind.iloc[-1]
        try:
            last_price = float(last_row['Close'])
            st.metric(label=f"{t} Last Close (USD)", value=f"${last_price:,.6f}")
        except Exception:
            st.metric(label=f"{t} Last Close (USD)", value="N/A")
        signal = generate_signal(df_ind)
        st.markdown(f"**Signal:** `{signal}`")
        # show small table
        st.dataframe(df_ind.tail(8).reset_index().rename(columns={df_ind.index.name or 'index': 'Date'}))
        # forecast with Prophet
        forecast_df = None
        if enable_prophet:
            try:
                forecast_df = run_prophet(close_ser, periods=forecast_choice)
            except Exception as e:
                st.warning(f"Prophet not available for {t}: {e}")
                forecast_df = None
        # plot interactive
        fig = plot_series_with_indicators(df_ind, forecast_df, title=f"{t} Price & Forecast")
        st.plotly_chart(fig, use_container_width=True)
        # download forecast if exists
        if forecast_df is not None:
            csv_bytes = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(f"Download {t} forecast CSV", csv_bytes, file_name=f"{t}_forecast.csv", mime='text/csv')

# combined summary download
if download_summary and results:
    rows = []
    for k, df_ind in results.items():
        tail = df_ind.tail(5).reset_index().rename(columns={df_ind.index.name or 'index': 'Date'})
        tail['ticker'] = k
        rows.append(tail[['ticker', 'Date', 'Close', 'MA20', 'MA50']])
    if rows:
        combined = pd.concat(rows, ignore_index=True)
        st.download_button("Download combined summary CSV", combined.to_csv(index=False).encode('utf-8'), file_name='summary_last_rows.csv', mime='text/csv')

# news section
if show_news:
    st.markdown("---")
    st.header("📢 News")
    try:
        news_items = fetch_news(limit=10)
        if not news_items:
            st.info("No news items found (check data/sample_news.csv or config/settings.json for API key).")
        else:
            news_df = pd.DataFrame(news_items)
            st.dataframe(news_df)
    except Exception as e:
        st.error(f"News fetch error: {e}")

st.markdown("---")
st.caption("Made with ❤️ — CryptoForecast. Not financial advice.")
