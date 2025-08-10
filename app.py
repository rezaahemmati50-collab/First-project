# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import date

# ---------- تنظیمات کلی ----------
st.set_page_config(page_title="CryptoForecast — Ultimate", layout="wide")
st.title("📊 CryptoForecast — داشبورد تحلیل و پیش‌بینی (نسخه نهایی)")

# ---------- توابع کمکی ----------
def compute_sma(series, window):
    return series.rolling(window=window).mean()

def compute_rsi(series, period=14):
    # ساده و ایمن: اگر سری طول کافی نداشت، NaN می‌دهد
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=(period-1), min_periods=period).mean()
    ma_down = down.ewm(com=(period-1), min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def safe_to_numeric(col):
    # چک می‌کند col یک سری/لیست است و سپس تبدیل می‌کند
    if isinstance(col, (pd.Series, list, tuple, np.ndarray)):
        return pd.to_numeric(col, errors='coerce')
    else:
        # تلاش برای تبدیل DataFrame تک‌ستونه
        try:
            s = pd.Series(col)
            return pd.to_numeric(s, errors='coerce')
        except Exception:
            return pd.Series(dtype='float64')

def extract_close_series(df, primary_ticker=None):
    # returns pd.Series or (None, message)
    if df is None or df.empty:
        return None, "دیتافریم خالی است."
    # MultiIndex columns (multi tickers)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(1):
            try:
                close_df = df.xs('Close', axis=1, level=1)
                # اگر تیکر اصلی مشخص است استفاده کن
                if primary_ticker and primary_ticker in close_df.columns:
                    return close_df[primary_ticker].copy(), f"Close for {primary_ticker} selected."
                # در غیر اینصورت اولین ستون را انتخاب کن
                return close_df.iloc[:, 0].copy(), "Multiple Close columns found; using first."
            except Exception as e:
                return None, f"Error extracting Close from MultiIndex: {e}"
        else:
            return None, "MultiIndex ولی سطح 'Close' موجود نیست."
    # عادی
    if 'Close' in df.columns:
        close_col = df['Close']
        if isinstance(close_col, pd.Series):
            return close_col.copy(), "Close column as Series."
        if isinstance(close_col, pd.DataFrame) and close_col.shape[1] >= 1:
            if primary_ticker and primary_ticker in close_col.columns:
                return close_col[primary_ticker].copy(), f"Close for {primary_ticker} taken."
            return close_col.iloc[:, 0].copy(), "df['Close'] is DataFrame; using first column."
        # fallback
        try:
            s = pd.Series(close_col)
            return s, "Close converted to Series."
        except Exception:
            return None, f"Unexpected Close type: {type(close_col)}"
    # try 'Adj Close'
    if 'Adj Close' in df.columns:
        adj = df['Adj Close']
        if isinstance(adj, pd.Series):
            return adj.copy(), "Using 'Adj Close' as fallback."
        if isinstance(adj, pd.DataFrame) and adj.shape[1] >= 1:
            return adj.iloc[:,0].copy(), "Using first column of 'Adj Close'."
    return None, "No 'Close' or 'Adj Close' found."

# ---------- بخش Sidebar (تنظیمات) ----------
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_input("Tickers (comma separated) - example: BTC-USD, ETH-USD, AAPL", "BTC-USD")
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
period = st.sidebar.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
predict_days = st.sidebar.slider("Forecast horizon (days)", 3, 90, 30)
use_prophet = st.sidebar.checkbox("Enable Prophet forecasting", value=True)
upload_csv = st.sidebar.file_uploader("Upload CSV (optional, columns ds,y or Date,Close)", type=["csv"])

# ---------- بارگذاری/تهیه دیتا ----------
@st.cache_data(ttl=300)
def download_yf(ticker, period, interval):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        # yfinance may return empty or multiindex
        return df
    except Exception as e:
        return pd.DataFrame()

# اگر فایل آپلود شده، از آن استفاده کن (پشتیبانی از ستون ds,y یا Date,Close)
def load_uploaded_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        # normalize column names
        cols = [c.strip().lower() for c in df.columns]
        if 'ds' in cols and 'y' in cols:
            df = df.rename(columns={df.columns[cols.index('ds')]: 'ds', df.columns[cols.index('y')]: 'y'})
            df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df = df.dropna(subset=['ds','y'])
            df = df.set_index('ds')
            return df
        # try Date,Close
        if 'date' in cols and 'close' in cols:
            df = df.rename(columns={df.columns[cols.index('date')]: 'Date', df.columns[cols.index('close')]: 'Close'})
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Date','Close'])
            df = df.set_index('Date')
            return df
    except Exception:
        pass
    return pd.DataFrame()

# If user uploaded CSV, prefer that
if upload_csv:
    user_df = load_uploaded_csv(upload_csv := upload_csv)
    if user_df is None or user_df.empty:
        st.sidebar.warning("Uploaded CSV couldn't be parsed as expected. Will try yfinance instead.")
        upload_csv = None
    else:
        st.sidebar.success("Using uploaded CSV as data source.")
else:
    user_df = None

# ---------- Main UI ----------
st.markdown("## Input & Data")
left, right = st.columns([2,1])

with left:
    st.write("**Tickers to analyze:**", tickers_input)
    st.write(f"**History:** {period} · **Interval:** {interval} · **Forecast:** {predict_days} days")
    manual_refresh = st.button("🔄 Refresh Data")

with right:
    st.write("Data source: Yahoo Finance" + (" + Uploaded CSV" if upload_csv else ""))

# parse tickers
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# container for combined chart
st.markdown("---")
st.header("📈 Combined Price Chart")
combined_fig = go.Figure()

# We'll collect results per ticker for display / download
results = {}

for t in tickers:
    # get data (uploaded CSV used only if single ticker and user expects local data)
    if user_df is not None and len(tickers)==1:
        df = user_df.copy()
        source_msg = "uploaded CSV"
    else:
        df = download_yf(t, period, interval)
        source_msg = "yfinance"

    st.write(f"### 🔹 {t}  — source: {source_msg}")

    if not isinstance(df, pd.DataFrame) or df.empty:
        st.error(f"❌ No data returned for {t}. Check ticker or period.")
        continue

    # ensure index/date column
    if 'Date' in df.columns:
        df = df.set_index('Date')

    # extract close series robustly
    close_series, msg = extract_close_series(df, primary_ticker=t)
    st.write(f"*debug:* {msg}")
    if close_series is None:
        st.error(f"❌ Could not extract Close for {t}. Skipping.")
        continue

    # ensure series type and numeric
    close_series = safe_to_numeric(close_series)
    if close_series.isna().all():
        st.error(f"❌ All Close values are non-numeric for {t}.")
        continue

    # align index: if df had datetime index, use it; else try series index
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
    else:
        # if close_series has own index, try to use it; else create range index
        try:
            idx = pd.to_datetime(close_series.index)
        except Exception:
            idx = pd.date_range(end=date.today(), periods=len(close_series))

    # construct a clean DataFrame
    data = pd.DataFrame({'Close': close_series.values}, index=idx)
    data = data.dropna(subset=['Close'])
    if data.empty:
        st.error(f"❌ After cleaning no data left for {t}.")
        continue

    # compute indicators
    data['MA20'] = compute_sma(data['Close'], 20)
    data['MA50'] = compute_sma(data['Close'], 50)
    data['MA200'] = compute_sma(data['Close'], 200)
    data['RSI14'] = compute_rsi(data['Close'], period=14)

    # add to combined chart
    combined_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name=f"{t} Close"))

    # Show small summary metrics
    last_price = float(data['Close'].iloc[-1])
    change_24h = None
    try:
        prev = data['Close'].iloc[-2]
        change_24h = (last_price - prev) / prev * 100
    except Exception:
        change_24h = 0.0

    st.metric(label="آخرین قیمت (USD)", value=f"${last_price:,.4f}", delta=f"{change_24h:.2f}%")

    # show mini chart & table
    st.plotly_chart(
        go.Figure([
            go.Scatter(x=data.index, y=data['Close'], name='Close'),
            go.Scatter(x=data.index, y=data['MA20'], name='MA20'),
            go.Scatter(x=data.index, y=data['MA50'], name='MA50'),
        ]).update_layout(title=f"{t} — Price & Moving Averages", xaxis_rangeslider_visible=True),
        use_container_width=True
    )

    st.subheader("📋 Recent Data & Indicators")
    st.dataframe(data.tail(10))

    # Simple rule-based signal
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

    # Forecast with Prophet (optional)
    if use_prophet:
        st.subheader(f"🔮 Prophet forecast for {t} ({predict_days} days)")
        # prepare dataframe for prophet
        prophet_df = pd.DataFrame({'ds': data.index, 'y': data['Close'].values})
        prophet_df = prophet_df.dropna()
        if prophet_df.shape[0] < 2:
            st.warning("⚠️ Not enough rows for Prophet forecasting.")
        else:
            try:
                m = Prophet(daily_seasonality=True)
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=predict_days)
                forecast = m.predict(future)

                # plot interactive
                fig_forecast = plot_plotly(m, forecast)
                st.plotly_chart(fig_forecast, use_container_width=True)

                # show forecast tail and download
                st.dataframe(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(10))
                csv = forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv(index=False).encode('utf-8')
                st.download_button(label=f"Download forecast CSV for {t}", data=csv, file_name=f"{t}_forecast.csv", mime='text/csv')
                results[t] = {'data': data, 'forecast': forecast}
            except Exception as e:
                st.error(f"❌ Prophet error for {t}: {e}")
    else:
        results[t] = {'data': data, 'forecast': None}

st.markdown("---")
st.header("📊 Combined chart (all selected) ")
combined_fig.update_layout(title="Combined Close Series", xaxis_rangeslider_visible=True, height=500)
st.plotly_chart(combined_fig, use_container_width=True)

# Summary download: zipped CSVs could be implemented; here offer concatenated last rows
if results:
    all_last = []
    for k,v in results.items():
        d = v['data'].copy()
        d = d.reset_index().rename(columns={'index':'ds', 'Close': 'Close'})
        d['ticker'] = k
        all_last.append(d[['ticker','ds','Close']].tail(5))
    all_concat = pd.concat(all_last, ignore_index=True)
    csv_all = all_concat.to_csv(index=False).encode('utf-8')
    st.download_button("Download summary CSV (last rows)", csv_all, file_name="summary_last_rows.csv", mime='text/csv')

st.caption("Made with ❤️ — CryptoForecast. Ensure proper usage: forecasts are informational, not financial advice.")
