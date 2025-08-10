# app.py - CryptoForecast (final integrated)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import datetime, date
import os

# ---------- Page config ----------
st.set_page_config(page_title="CryptoForecast — Final", layout="wide")
st.title("📊 CryptoForecast — داشبورد نهایی تحلیل و پیش‌بینی")

# ---------- Helpers ----------
def compute_sma(s, w):
    return s.rolling(window=w).mean()

def compute_rsi(s, period=14):
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    # use simple EMA smoothing to avoid NANs quickly
    ma_up = up.ewm(span=period, adjust=False, min_periods=period).mean()
    ma_down = down.ewm(span=period, adjust=False, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def safe_series(x):
    # convert different types to pd.Series
    if isinstance(x, pd.Series):
        return x
    try:
        return pd.Series(x)
    except Exception:
        return pd.Series(dtype='float64')

def safe_to_numeric(col):
    s = safe_series(col)
    return pd.to_numeric(s, errors='coerce')

def extract_close_series(df, primary_ticker=None):
    """
    Robust extracting of Close series from yfinance output.
    Returns (series_or_None, message)
    """
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return None, "DataFrame is empty."

    # If yfinance returned MultiIndex columns (multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        # try to extract level 'Close'
        try:
            if 'Close' in df.columns.get_level_values(1):
                close_df = df.xs('Close', axis=1, level=1)
                # If primary ticker specified and exists
                if primary_ticker and primary_ticker in close_df.columns:
                    return close_df[primary_ticker].copy(), f"Close for {primary_ticker} selected."
                # else take first available
                return close_df.iloc[:, 0].copy(), f"Multiple Close columns; using first ({close_df.columns[0]})."
            else:
                return None, "MultiIndex present but no 'Close' level found."
        except Exception as e:
            return None, f"Error extracting Close from MultiIndex: {e}"

    # Normal case
    if 'Close' in df.columns:
        close_col = df['Close']
        if isinstance(close_col, pd.Series):
            return close_col.copy(), "Close Series extracted."
        if isinstance(close_col, pd.DataFrame) and close_col.shape[1] > 0:
            # data like df['Close'] returns DataFrame (rare)
            if primary_ticker and primary_ticker in close_col.columns:
                return close_col[primary_ticker].copy(), f"Close for {primary_ticker} from df['Close']."
            return close_col.iloc[:,0].copy(), "df['Close'] is DataFrame; using first column."
        try:
            s = pd.Series(close_col)
            return s, "Close converted to Series."
        except Exception:
            return None, f"Unexpected Close type: {type(close_col)}"

    # fallback to 'Adj Close'
    if 'Adj Close' in df.columns:
        adj = df['Adj Close']
        if isinstance(adj, pd.Series):
            return adj.copy(), "Using 'Adj Close' fallback."
        if isinstance(adj, pd.DataFrame) and adj.shape[1]>0:
            return adj.iloc[:,0].copy(), "Using first column of 'Adj Close'."

    return None, "No Close or Adj Close found."

@st.cache_data(ttl=300)
def download_data_yf(ticker, period="1y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        # often df empty or multiindex; return as-is
        return df
    except Exception:
        return pd.DataFrame()

def load_uploaded_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return pd.DataFrame()
    cols = [c.strip().lower() for c in df.columns]
    # ds,y
    if 'ds' in cols and 'y' in cols:
        df = df.rename(columns={df.columns[cols.index('ds')]: 'ds', df.columns[cols.index('y')]: 'y'})
        df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df = df.dropna(subset=['ds','y']).set_index('ds')
        return df
    if 'date' in cols and 'close' in cols:
        df = df.rename(columns={df.columns[cols.index('date')]: 'Date', df.columns[cols.index('close')]: 'Close'})
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Date','Close']).set_index('Date')
        return df
    return pd.DataFrame()

# ---------- Sidebar ----------
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_input("Tickers (comma separated). Example: BTC-USD, ETH-USD", value="BTC-USD")
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y","2y","5y","max"], index=3)
interval = st.sidebar.selectbox("Interval", ["1d","1wk","1mo"], index=0)
forecast_days = st.sidebar.slider("Forecast horizon (days)", 3, 90, 30)
enable_prophet = st.sidebar.checkbox("Enable Prophet forecasting", value=True)
upload_file = st.sidebar.file_uploader("Upload CSV (optional) - formats: ds,y  or Date,Close", type=["csv"])
show_news = st.sidebar.checkbox("Show News (sample)", value=True)
auto_refresh = st.sidebar.checkbox("Auto-refresh data (manual rerun recommended)", value=False)

# ---------- Data source selection ----------
user_df = None
if upload_file:
    user_df = load_uploaded_csv(upload_file)
    if user_df is None or user_df.empty:
        st.sidebar.warning("Uploaded CSV couldn't be parsed. Will fallback to yfinance.")
        user_df = None
    else:
        st.sidebar.success("Using uploaded CSV as data source.")

# ---------- Main layout ----------
st.markdown("### Inputs")
col1, col2 = st.columns([3,1])
with col1:
    st.write("Tickers:", tickers_input)
    st.write(f"History: {period} · Interval: {interval} · Forecast: {forecast_days} days")
with col2:
    if st.button("🔄 Refresh Now"):
        # simple control; cached functions will still respect ttl
        st.experimental_rerun()

tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

st.markdown("---")
st.header("📈 Price & Indicators")

combined_fig = go.Figure()
results = {}

for t in tickers:
    st.subheader(f"🔹 {t}")
    # load data
    if user_df is not None and len(tickers)==1:
        df_raw = user_df.copy()
        source_msg = "uploaded CSV"
    else:
        df_raw = download_data_yf(t, period=period, interval=interval)
        source_msg = "yfinance"

    st.write(f"*Source:* {source_msg}")

    if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        st.error(f"❌ No data returned for {t}. Check ticker and period.")
        continue

    # If yfinance returned DataFrame with columns including Date, but index may be DatetimeIndex
    if 'Date' in df_raw.columns:
        try:
            df_raw = df_raw.set_index('Date')
        except Exception:
            pass

    # extract Close series robustly
    close_ser, msg = extract_close_series(df_raw, primary_ticker=t)
    st.write(f"*debug:* {msg}")
    if close_ser is None:
        st.error(f"❌ Could not extract Close for {t}. Skipping.")
        continue

    # convert to numeric and drop NaNs
    close_ser = safe_to_numeric(close_ser)
    close_ser = close_ser.dropna()
    if close_ser.empty:
        st.error(f"❌ After cleaning, no numeric Close data for {t}.")
        continue

    # determine index (dates)
    if isinstance(df_raw.index, pd.DatetimeIndex):
        idx = df_raw.index
        # if lengths mismatch, try to align by length
        if len(idx) != len(close_ser):
            # try to use close_ser.index if datetime
            try:
                idx = pd.to_datetime(close_ser.index)
            except Exception:
                idx = pd.date_range(end=date.today(), periods=len(close_ser))
    else:
        try:
            idx = pd.to_datetime(close_ser.index)
        except Exception:
            idx = pd.date_range(end=date.today(), periods=len(close_ser))

    data = pd.DataFrame({'Close': close_ser.values}, index=idx)
    data = data.dropna(subset=['Close'])
    if data.empty:
        st.error(f"❌ No valid rows for {t} after building DataFrame.")
        continue

    # indicators
    data['MA20'] = compute_sma(data['Close'], 20)
    data['MA50'] = compute_sma(data['Close'], 50)
    data['MA200'] = compute_sma(data['Close'], 200)
    data['RSI14'] = compute_rsi(data['Close'], period=14)

    # combined chart add
    combined_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name=f"{t}"))

    # show metrics
    last_price = float(data['Close'].iloc[-1])
    delta_pct = 0.0
    if len(data['Close']) >= 2:
        prev = data['Close'].iloc[-2]
        try:
            delta_pct = (last_price - prev)/prev * 100
        except Exception:
            delta_pct = 0.0
    st.metric(label="آخرین قیمت (USD)", value=f"${last_price:,.4f}", delta=f"{delta_pct:.2f}%")

    # small chart for this ticker
    fig_mini = go.Figure()
    fig_mini.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close'))
    fig_mini.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20'))
    fig_mini.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50'))
    fig_mini.update_layout(title=f"{t} — Price & MA", xaxis_rangeslider_visible=True, height=320)
    st.plotly_chart(fig_mini, use_container_width=True)

    st.subheader("📋 Recent data & indicators")
    st.dataframe(data.tail(8))

    # simple rule signal
    signal = "🟡 Hold"
    try:
        if data['RSI14'].iloc[-1] < 30 and data['MA20'].iloc[-1] > data['MA50'].iloc[-1]:
            signal = "🔵 Strong Buy"
        elif data['RSI14'].iloc[-1] > 70 and data['MA20'].iloc[-1] < data['MA50'].iloc[-1]:
            signal = "🔴 Strong Sell"
        elif data['MA20'].iloc[-1] > data['MA50'].iloc[-1]:
            signal = "🟢 Buy"
        elif data['MA20'].iloc[-1] < data['MA50'].iloc[-1]:
            signal = "🔴 Sell"
    except Exception:
        signal = "🟡 Hold"

    st.markdown(f"### Signal: {signal}")

    # Prophet forecast block
    forecast = None
    if enable_prophet:
        st.subheader(f"🔮 Prophet forecast for {t} ({forecast_days} days)")
        prophet_df = pd.DataFrame({'ds': data.index, 'y': data['Close'].values}).dropna()
        if prophet_df.shape[0] < 2:
            st.warning("⚠️ Not enough rows for Prophet forecasting.")
        else:
            try:
                m = Prophet(daily_seasonality=True)
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=forecast_days)
                forecast = m.predict(future)
                st.plotly_chart(plot_plotly(m, forecast), use_container_width=True)
                st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(8))
                csv_bytes = forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv(index=False).encode('utf-8')
                st.download_button(label=f"Download forecast CSV for {t}", data=csv_bytes, file_name=f"{t}_forecast.csv", mime='text/csv')
            except Exception as e:
                st.error(f"❌ Prophet error for {t}: {e}")

    results[t] = {'data': data, 'forecast': forecast}

# After loop - combined chart
st.markdown("---")
st.header("📊 Combined chart")
combined_fig.update_layout(title="Combined Close Series", xaxis_rangeslider_visible=True, height=450)
st.plotly_chart(combined_fig, use_container_width=True)

# Summary download
if results:
    rows = []
    for k,v in results.items():
        d = v['data'].reset_index().rename(columns={'index':'ds', 'Close':'Close'})
        d['ticker'] = k
        rows.append(d[['ticker','ds','Close']].tail(5))
    combined_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not combined_df.empty:
        st.download_button("Download summary (last rows) CSV", combined_df.to_csv(index=False).encode('utf-8'), file_name="summary_last_rows.csv", mime='text/csv')

# News section (local sample)
if show_news:
    st.markdown("---")
    st.header("📰 News (sample)")
    news_path = os.path.join("data","sample_news.csv")
    if os.path.exists(news_path):
        try:
            news_df = pd.read_csv(news_path)
            st.dataframe(news_df.head(10))
        except Exception:
            st.info("News file exists but couldn't be read.")
    else:
        st.info("No local news file found. You can place sample_news.csv in data/ folder.")

st.caption("Made with ❤️ — CryptoForecast. Not financial advice. Use carefully.")
