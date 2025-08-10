# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Forecast + Prophet", layout="wide")
st.title("📊 داشبورد تحلیل و پیش‌بینی کریپتو")

# Try to import Prophet; if unavailable we'll fallback
HAS_PROPHET = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

# ---------------- UI: controls ----------------
col1, col2, col3 = st.columns([2,1,1])
with col1:
    symbol = st.selectbox("انتخاب ارز دیجیتال:", ["ETH-USD","BTC-USD","ADA-USD","SOL-USD","XRP-USD"], index=0)
with col2:
    period = st.selectbox("دوره داده:", ["3mo","6mo","1y","2y"], index=1)
with col3:
    interval = st.selectbox("فواصل زمانی:", ["1d","1wk"], index=0)

st.markdown("---")

# Forecast controls
fc_col1, fc_col2 = st.columns([2,1])
with fc_col1:
    forecast_days = st.slider("تعداد روزهای پیش‌بینی:", 1, 90, 14)
with fc_col2:
    use_log = st.checkbox("استفاده از log-transform (پیشنهادی)", value=True)

st.markdown(f"**Prophet در دسترس است:** {'✅' if HAS_PROPHET else '❌ (fallback → Moving Average)'}")
st.markdown("---")

# ---------------- Helpers ----------------
@st.cache_data(ttl=300)
def download_data(sym, period, interval):
    df = yf.download(sym, period=period, interval=interval, progress=False)
    return df

def normalize_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has 'Date' column and 'Close' numeric column.
    Raises ValueError if cannot find Close.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    df = raw_df.copy()
    # if datetime index -> reset
    if np.issubdtype(df.index.dtype, np.datetime64):
        df = df.reset_index()
    # find date column
    date_col = None
    for c in df.columns:
        if str(c).lower() in ("date","datetime","index"):
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    # find close column (case-insensitive)
    close_col = None
    for c in df.columns:
        if str(c).lower() == "close":
            close_col = c
            break
    if close_col is None:
        for c in df.columns:
            if str(c).lower() in ("adj close","adjusted close"):
                close_col = c
                break
    if close_col is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric:
            close_col = numeric[-1]
    if close_col is None:
        raise ValueError("ستون 'Close' یافت نشد. لطفاً CSV با ستون Date و Close آپلود کنید یا نماد دیگری را انتخاب کنید.")
    df = df.rename(columns={close_col: "Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    return df

def moving_avg_forecast(series: pd.Series, days: int):
    last = float(series.iloc[-1])
    avg_pct = series.pct_change().dropna().mean()
    # if avg_pct is NaN (constant series) fallback to 0
    if np.isnan(avg_pct):
        avg_pct = 0.0
    return [ last * ((1+avg_pct)**i) for i in range(1, days+1) ]

def to_csv_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")

# ---------------- Load data (yfinance) ----------------
with st.spinner("در حال دریافت داده‌ها ..."):
    raw = download_data(symbol, period, interval)

if raw is None or raw.empty:
    st.error("داده‌ای دریافت نشد — لطفاً اتصال اینترنت یا نماد/بازه را بررسی کنید.")
    st.stop()

# normalize and validate
try:
    df = normalize_df(raw)
except Exception as e:
    st.error(f"Data normalization error: {e}")
    st.stop()

# ---------------- Show last data ----------------
st.subheader("داده‌های تاریخی (آخرین مشاهدات)")
st.dataframe(df[["Date","Close"]].tail(10), use_container_width=True)

# ---------------- Historical price chart ----------------
st.subheader("نمودار قیمت تاریخی")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close",
                         line=dict(color="#00BCD4", width=2)))
fig.update_layout(template="plotly_dark", height=420, margin=dict(t=30,b=10), xaxis_title="", yaxis_title="Price (quote)")
st.plotly_chart(fig, use_container_width=True)

# ---------------- Moving Average (user) ----------------
ma_window = st.slider("پنجره میانگین متحرک (روز)", 5, 100, 20)
df["MA"] = df["Close"].rolling(window=ma_window).mean()

st.subheader(f"میانگین متحرک {ma_window}-روزه")
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close", line=dict(color="#BBBBBB")))
fig_ma.add_trace(go.Scatter(x=df["Date"], y=df["MA"], mode="lines", name=f"MA{ma_window}", line=dict(color="#FFA500", width=2)))
fig_ma.update_layout(template="plotly_dark", height=420, margin=dict(t=30,b=10))
st.plotly_chart(fig_ma, use_container_width=True)

# ---------------- Forecast ----------------
st.subheader(f"🔮 پیش‌بینی {forecast_days} روز آینده")

future_dates = [ df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days+1) ]
forecast_vals = None
forecast_df = None

if HAS_PROPHET:
    try:
        # prepare df for prophet
        p_df = df[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"}).copy()
        log_used = False
        if use_log and (p_df["y"]>0).all():
            p_df["y"] = np.log(p_df["y"])
            log_used = True
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, changepoint_prior_scale=0.05)
        m.fit(p_df.rename(columns={"y":"y"}))
        future = m.make_future_dataframe(periods=forecast_days, freq='D')
        pred = m.predict(future)
        if log_used:
            pred["yhat_final"] = np.exp(pred["yhat"])
            pred["yhat_lower_final"] = np.exp(pred["yhat_lower"])
            pred["yhat_upper_final"] = np.exp(pred["yhat_upper"])
        else:
            pred["yhat_final"] = pred["yhat"]
            pred["yhat_lower_final"] = pred["yhat_lower"]
            pred["yhat_upper_final"] = pred["yhat_upper"]
        # take last forecast_days rows
        fvals = pred[["ds","yhat_final","yhat_lower_final","yhat_upper_final"]].tail(forecast_days).copy()
        forecast_vals = fvals["yhat_final"].values
        forecast_df = fvals.rename(columns={"ds":"Date","yhat_final":"yhat","yhat_lower_final":"yhat_lower","yhat_upper_final":"yhat_upper"})
        # convert Date to date
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.date
    except Exception as e:
        st.warning(f"خطا در اجرای Prophet: {e}\nاز روش میانگین متحرک استفاده می‌شود.")
        forecast_vals = np.array(moving_avg_forecast(df["Close"], forecast_days))
        forecast_df = pd.DataFrame({"Date":[d.date() for d in future_dates], "yhat":np.round(forecast_vals,2)})
else:
    # fallback: moving average forecast
    forecast_vals = np.array(moving_avg_forecast(df["Close"], forecast_days))
    forecast_df = pd.DataFrame({"Date":[d.date() for d in future_dates], "yhat":np.round(forecast_vals,2)})

# ensure forecast_vals is 1-d array
forecast_vals = np.asarray(forecast_vals).reshape(-1,)

# ---------------- Forecast table & downloads ----------------
st.markdown("### جدول پیش‌بینی")
st.dataframe(forecast_df.reset_index(drop=True), use_container_width=True)
st.download_button("دانلود پیش‌بینی (CSV)", to_csv_bytes(forecast_df), file_name=f"{symbol}_forecast.csv")

# ---------------- Forecast plot (actual + forecast + uncertainty if present) ----------------
st.subheader("نمودار پیش‌بینی (خط پیش‌بینی و ناحیه عدم قطعیت)")
fig_f = go.Figure()
fig_f.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Actual", line=dict(color="#9AA0A6")))

# forecast line
fig_f.add_trace(go.Scatter(x=future_dates, y=forecast_vals, mode="lines+markers", name="Forecast",
                           line=dict(color="#FFA500", width=2, dash="dot")))

# uncertainty band if available
if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
    lower = forecast_df["yhat_lower"].values
    upper = forecast_df["yhat_upper"].values
    xs = list(pd.to_datetime(forecast_df["Date"])) + list(pd.to_datetime(forecast_df["Date"])[::-1])
    ys = list(upper) + list(lower[::-1])
    fig_f.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor="rgba(255,165,0,0.12)",
                               line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=False))

fig_f.update_layout(template="plotly_dark", height=460, margin=dict(t=30,b=10))
st.plotly_chart(fig_f, use_container_width=True)

st.markdown("---")
st.markdown("**تذکر:** پیش‌بینی‌ها تنها بر اساس داده‌های تاریخی محاسبه می‌شوند و به هیچ وجه توصیهٔ سرمایه‌گذاری نیستند.")
