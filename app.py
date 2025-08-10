# app.py — Product-ready Crypto Forecast Dashboard (Persian UI)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="AureumAI · Final Dashboard", layout="wide")
st.title("AureumAI · داشبورد نهایی تحلیل و پیش‌بینی کریپتو")

# Try optional Prophet import (fallback to simple forecast if not available)
HAS_PROPHET = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("⚙️ تنظیمات")

ASSETS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "Solana (SOL)": "SOL-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Dogecoin (DOGE)": "DOGE-USD"
}
asset_name = st.sidebar.selectbox("انتخاب ارز", list(ASSETS.keys()), index=1)
symbol = ASSETS[asset_name]

period = st.sidebar.selectbox("بازهٔ تاریخی برای آموزش", ["3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("فاصلهٔ داده", ["1d","1wk"], index=0)

forecast_days = st.sidebar.slider("تعداد روزهای پیش‌بینی", 7, 90, 30)
model_choice = st.sidebar.selectbox("مدل پیش‌بینی (اگر Prophet نصب نیست به Moving Avg برمی‌گردد)",
                                    [ "Auto (Prophet → MovingAvg)", "Prophet (if available)", "MovingAvg (fast)" ])
use_log = st.sidebar.checkbox("استفاده از log-transform (پیشنهادی)", value=True)
st.sidebar.markdown("---")
st.sidebar.write(f"Prophet در دسترس: {'✔' if HAS_PROPHET else '✖'}")
st.sidebar.markdown("AureumAI · Demo — این ابزار مشاور سرمایه‌گذاری نیست.")

# ---------------------------
# Helper utilities
# ---------------------------
@st.cache_data(ttl=300)
def fetch_data(sym, period, interval):
    df = yf.download(sym, period=period, interval=interval, progress=False)
    return df

def normalize_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    df = raw_df.copy()
    # bring index to column if datetime index
    if np.issubdtype(df.index.dtype, np.datetime64):
        df = df.reset_index()
    # find date col
    date_col = None
    for c in df.columns:
        if str(c).lower() in ("date","datetime","index"):
            date_col = c; break
    if date_col is None:
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    # find close col
    close_col = None
    for c in df.columns:
        if str(c).lower() == "close":
            close_col = c; break
    if close_col is None:
        for c in df.columns:
            if str(c).lower() in ("adj close","adjusted close"):
                close_col = c; break
    if close_col is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric:
            close_col = numeric[-1]
    if close_col is None:
        raise ValueError("ستون Close یافت نشد — لطفاً CSV با ستون Date و Close آپلود کنید.")
    df = df.rename(columns={close_col: "Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    return df

def moving_avg_forecast(series: pd.Series, days: int):
    last = float(series.iloc[-1])
    avg_pct = series.pct_change().dropna().mean()
    return [ last * ((1 + avg_pct) ** i) for i in range(1, days+1) ]

def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# ---------------------------
# Load data (yfinance) or allow CSV upload
# ---------------------------
uploaded = st.sidebar.file_uploader("آپلود CSV (اختیاری) — ستون Date و Close", type=["csv"])
if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        st.sidebar.success("CSV آپلود شد — از آن استفاده می‌کنیم.")
    except Exception as e:
        st.sidebar.error(f"خطا در خواندن CSV: {e}")
        raw = pd.DataFrame()
else:
    with st.spinner(f"دریافت داده {symbol} ..."):
        raw = fetch_data(symbol, period=period, interval=interval)

if raw is None or raw.empty:
    st.error("داده‌ای در دسترس نیست. اتصال اینترنت را بررسی کنید یا فایل CSV آپلود کنید.")
    st.stop()

# normalize
try:
    df = normalize_df(raw)
except Exception as e:
    st.error(f"Data normalization error: {e}")
    st.stop()

# ---------------------------
# Main UI: Tabs
# ---------------------------
st.markdown(f"### {asset_name} — `{symbol}`")
st.markdown(f"تاریخ داده‌ها: {df['Date'].min().date()} → {df['Date'].max().date()}  •  سطرها: {len(df)}")

tab1, tab2, tab3, tab4 = st.tabs(["📈 Price", "📊 Data", "🔮 Forecast", "📉 Technical"])

# Price tab
with tab1:
    st.subheader("نمودار قیمت تاریخی")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close", line=dict(color="#4CAF50")))
    fig.update_layout(template="plotly_dark", margin=dict(t=30,b=10), height=420, xaxis_title="", yaxis_title="Price (quote)")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("قیمت آخر (Close)", f"${df['Close'].iloc[-1]:,.2f}")

# Data tab
with tab2:
    st.subheader("دادهٔ خام (آخرین ۲۰ سطر)")
    st.dataframe(df.tail(20), use_container_width=True)
    st.download_button("دانلود دادهٔ خام (CSV)", to_csv_bytes(df), file_name=f"{symbol}_raw.csv")

# Forecast tab
with tab3:
    st.subheader("پیش‌بینی")
    chosen = model_choice
    if model_choice.startswith("Auto"):
        chosen = "Prophet" if HAS_PROPHET else "MovingAvg"
    st.info(f"مدل انتخاب‌شده: {chosen}")

    # prepare dataframe for prophet if used
    forecast_df = None
    future_dates = [ df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days+1) ]

    try:
        if chosen == "Prophet" and HAS_PROPHET:
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
            forecast_vals = pred["yhat_final"].tail(forecast_days).values
            forecast_df = pred[["ds","yhat_final","yhat_lower_final","yhat_upper_final"]].tail(forecast_days).rename(
                columns={"ds":"Date","yhat_final":"yhat","yhat_lower_final":"yhat_lower","yhat_upper_final":"yhat_upper"}
            )
        else:
            # Moving average fallback
            forecast_vals = np.array(moving_avg_forecast(df["Close"], forecast_days))
            forecast_df = pd.DataFrame({"Date":[d.date() for d in future_dates],"yhat":np.round(forecast_vals,2)})
    except Exception as e:
        st.warning(f"خطا در پیش‌بینی با مدل انتخابی: {e}\nاز MovingAvg استفاده می‌شود.")
        forecast_vals = np.array(moving_avg_forecast(df["Close"], forecast_days))
        forecast_df = pd.DataFrame({"Date":[d.date() for d in future_dates],"yhat":np.round(forecast_vals,2)})

    # show table and downloads
    st.markdown("#### جدول پیش‌بینی")
    st.dataframe(forecast_df.reset_index(drop=True), use_container_width=True)
    st.download_button("دانلود پیش‌بینی (CSV)", to_csv_bytes(forecast_df), file_name=f"{symbol}_forecast.csv")

    # plot forecast (actual + forecast)
    st.markdown("#### نمودار پیش‌بینی")
    fig_f = go.Figure()
    fig_f.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Actual", line=dict(color="#AAAAAA")))
    # forecast line
    fig_f.add_trace(go.Scatter(x=future_dates, y=forecast_vals, mode="lines+markers", name="Forecast", line=dict(color="#FFA500", dash="dot")))
    # uncertainty if available
    if "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
        fig_f.add_trace(go.Scatter(
            x=list(pd.to_datetime(forecast_df["Date"])) + list(pd.to_datetime(forecast_df["Date"])[::-1]),
            y=list(forecast_df["yhat_upper"]) + list(forecast_df["yhat_lower"][::-1]),
            fill="toself", fillcolor="rgba(200,200,200,0.12)", line=dict(color="rgba(255,255,255,0)"), showlegend=False
        ))
    fig_f.update_layout(template="plotly_dark", height=420, margin=dict(t=20,b=10))
    st.plotly_chart(fig_f, use_container_width=True)

# Technical tab
with tab4:
    st.subheader("Indicator: RSI & MACD (اختیاری)")
    try:
        import ta
        df_t = df.copy()
        df_t["RSI"] = ta.momentum.RSIIndicator(df_t["Close"], window=14).rsi()
        macd = ta.trend.MACD(df_t["Close"])
        df_t["MACD"] = macd.macd()
        df_t["MACD_signal"] = macd.macd_signal()

        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_t["Date"], y=df_t["RSI"], mode="lines", name="RSI"))
        fig_rsi.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_rsi, use_container_width=True)

        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df_t["Date"], y=df_t["MACD"], mode="lines", name="MACD"))
        fig_macd.add_trace(go.Scatter(x=df_t["Date"], y=df_t["MACD_signal"], mode="lines", name="Signal"))
        fig_macd.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_macd, use_container_width=True)
    except Exception:
        st.info("پکیج 'ta' نصب نیست. برای تحلیل تکنیکال (RSI/MACD) آن را نصب کنید: pip install ta")

st.markdown("---")
st.caption("AureumAI · Final Demo — Not financial advice. For production, secure credentials, logging and rate limits are required.")
