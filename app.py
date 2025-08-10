# app.py
# Persian UI — Single-default asset (BTC) + Forecast + Technical signals (RSI/MACD) + Safe fallbacks

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="AureumAI · BTC Forecast & Signal", layout="wide")
st.title("💹 AureumAI · پیش‌بینی و سیگنال (BTC پیش‌فرض)")

# Try optional libraries
HAS_PROPHET = False
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

HAS_TA = False
try:
    import ta
    HAS_TA = True
except Exception:
    HAS_TA = False

# -----------------------
# Sidebar: تنظیمات کلی
# -----------------------
st.sidebar.header("⚙️ تنظیمات")

ASSETS = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "Litecoin (LTC-USD)": "LTC-USD"
}

asset = st.sidebar.selectbox("انتخاب ارز (پیش‌فرض BTC):", list(ASSETS.keys()), index=0)
symbol = ASSETS[asset]

# بازه تاریخی و پیش‌بینی
period_choice = st.sidebar.selectbox("دادهٔ تاریخی برای آموزش:", ["3mo", "6mo", "1y", "2y"], index=1)
forecast_days = st.sidebar.slider("تعداد روزهای پیش‌بینی:", 1, 30, 7)
use_log = st.sidebar.checkbox("استفاده از log-transform در Prophet (اگر فعال است)", value=True)

st.sidebar.markdown("---")
st.sidebar.write(f"Prophet نصب شده: {'✔️' if HAS_PROPHET else '❌'}")
st.sidebar.write(f"پکیج ta نصب شده: {'✔️' if HAS_TA else '❌'}")
st.sidebar.markdown("AureumAI — این برنامه مشاورهٔ مالی نیست. تصمیم‌گیری شخصی لازم است.")

# -----------------------
# Helpers
# -----------------------
@st.cache_data(ttl=300)
def fetch_data(sym, period):
    try:
        df = yf.download(sym, period=period, interval="1d", progress=False)
        return df
    except Exception as e:
        return pd.DataFrame()

def normalize_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    df = raw_df.copy()
    # bring index to column if datetime index
    if np.issubdtype(df.index.dtype, np.datetime64):
        df = df.reset_index()
    # find date column
    date_col = None
    for c in df.columns:
        if str(c).lower() in ("date","datetime","index"):
            date_col = c; break
    if date_col is None:
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "Date"})
    # find close column
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
        raise ValueError("ستون 'Close' یافت نشد.")
    df = df.rename(columns={close_col: "Close"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    return df

def moving_avg_forecast(series: pd.Series, days: int):
    last = float(series.iloc[-1])
    avg_pct = series.pct_change().dropna().mean()
    if np.isnan(avg_pct):
        avg_pct = 0.0
    return [ last * ((1+avg_pct)**i) for i in range(1, days+1) ]

def compute_signals(df: pd.DataFrame, forecast_next: float = None):
    """
    Compute buy/sell/hold using RSI and MACD with simple rules:
    - RSI < 30  => oversold -> buy signal (positive)
    - RSI > 70  => overbought -> sell signal (negative)
    - MACD diff > 0 => bullish
    Combine indicators and forecast change:
    - if indicators bullish AND forecast increase -> Strong Buy
    - if indicators bearish AND forecast decrease -> Strong Sell
    - else Hold
    Returns (signal_text, reason)
    """
    reasons = []
    score = 0  # positive -> buy, negative -> sell

    # RSI (if available)
    if HAS_TA:
        try:
            rsi = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
            last_rsi = float(rsi.iloc[-1])
            if last_rsi < 30:
                score += 1
                reasons.append(f"RSI low ({last_rsi:.1f}) → oversold")
            elif last_rsi > 70:
                score -= 1
                reasons.append(f"RSI high ({last_rsi:.1f}) → overbought")
            else:
                reasons.append(f"RSI neutral ({last_rsi:.1f})")
        except Exception:
            reasons.append("RSI unavailable")
    else:
        reasons.append("RSI (ta) نصب نیست")

    # MACD (if available)
    if HAS_TA:
        try:
            macd = ta.trend.MACD(df["Close"])
            macd_diff = float((macd.macd_diff()).iloc[-1])
            if macd_diff > 0:
                score += 1
                reasons.append(f"MACD positive ({macd_diff:.4f})")
            elif macd_diff < 0:
                score -= 1
                reasons.append(f"MACD negative ({macd_diff:.4f})")
            else:
                reasons.append("MACD neutral")
        except Exception:
            reasons.append("MACD unavailable")
    else:
        reasons.append("MACD (ta) نصب نیست")

    # Forecast-based signal
    if forecast_next is not None:
        last_price = float(df["Close"].iloc[-1])
        # compute percent change
        pct = (forecast_next - last_price) / last_price if last_price != 0 else 0
        pct_percent = pct * 100
        if pct > 0.01:  # >1% up
            score += 1
            reasons.append(f"Forecast +{pct_percent:.2f}% → short-term up")
        elif pct < -0.01:
            score -= 1
            reasons.append(f"Forecast {pct_percent:.2f}% → short-term down")
        else:
            reasons.append(f"Forecast small change ({pct_percent:.2f}%)")

    # Decide final signal
    if score >= 2:
        return "🔵 سیگنال قوی خرید (Strong BUY)", " · ".join(reasons)
    elif score == 1:
        return "🟢 سیگنال خرید (BUY)", " · ".join(reasons)
    elif score == 0:
        return "🟡 نگه‌داری (HOLD)", " · ".join(reasons)
    elif score == -1:
        return "🔴 سیگنال فروش (SELL)", " · ".join(reasons)
    else:
        return "⚫ سیگنال قوی فروش (Strong SELL)", " · ".join(reasons)

# -----------------------
# Fetch and normalize data
# -----------------------
with st.spinner("در حال دریافت داده‌ها از Yahoo Finance ..."):
    raw = fetch_data(symbol, period_choice)

if raw is None or raw.empty:
    st.error("داده‌ای دریافت نشد — لطفاً اتصال اینترنت یا نماد/دوره را بررسی کنید.")
    st.stop()

try:
    df = normalize_df(raw)
except Exception as e:
    st.error(f"Data normalization error: {e}")
    st.stop()

# -----------------------
# Basic display
# -----------------------
st.subheader(f"📊 داده‌های تاریخی — {asset} ({symbol})")
st.write(f"تاریخچه از {df['Date'].min().date()} تا {df['Date'].max().date()} — {len(df)} ردیف")
st.dataframe(df[["Date","Close"]].tail(10), use_container_width=True)

# -----------------------
# Price chart
# -----------------------
st.subheader("📈 نمودار قیمت تاریخی")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["Date"].tolist(),
    y=df["Close"].tolist(),
    mode="lines",
    name="Close",
    line=dict(color="#00BFA6", width=2)
))
fig.update_layout(template="plotly_dark", height=420, margin=dict(t=30,b=10))
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Moving Average display
# -----------------------
ma_window = st.slider("پنجرهٔ میانگین متحرک (MA)", 5, 60, 20)
df["MA"] = df["Close"].rolling(window=ma_window).mean()
st.subheader(f"میانگین متحرک {ma_window}-روزه")
fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=df["Date"].tolist(), y=df["Close"].tolist(), mode="lines", name="Close", line=dict(color="#9AA0A6")))
fig_ma.add_trace(go.Scatter(x=df["Date"].tolist(), y=df["MA"].tolist(), mode="lines", name=f"MA{ma_window}", line=dict(color="#FFA500", width=2)))
fig_ma.update_layout(template="plotly_dark", height=360)
st.plotly_chart(fig_ma, use_container_width=True)

# -----------------------
# Forecast (Prophet if available)
# -----------------------
st.subheader(f"🔮 پیش‌بینی {forecast_days} روز آینده")
future_dates = [ df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days+1) ]
forecast_vals = None
forecast_df = None
used_prophet = False

if HAS_PROPHET:
    try:
        p_df = df[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"}).copy()
        log_used = False
        if use_log and (p_df["y"]>0).all():
            p_df["y"] = np.log(p_df["y"])
            log_used = True
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, changepoint_prior_scale=0.05)
        m.fit(p_df.rename(columns={'y':'y'}))
        future = m.make_future_dataframe(periods=forecast_days, freq='D')
        pred = m.predict(future)
        # revert log
        if log_used:
            pred["yhat_final"] = np.exp(pred["yhat"])
            pred["yhat_lower_final"] = np.exp(pred["yhat_lower"])
            pred["yhat_upper_final"] = np.exp(pred["yhat_upper"])
        else:
            pred["yhat_final"] = pred["yhat"]
            pred["yhat_lower_final"] = pred["yhat_lower"]
            pred["yhat_upper_final"] = pred["yhat_upper"]
        fvals = pred[["ds","yhat_final","yhat_lower_final","yhat_upper_final"]].tail(forecast_days).copy()
        # convert types
        forecast_vals = fvals["yhat_final"].values.astype(float)
        forecast_df = fvals.rename(columns={"ds":"Date","yhat_final":"yhat","yhat_lower_final":"yhat_lower","yhat_upper_final":"yhat_upper"})
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"]).dt.date
        used_prophet = True
    except Exception as e:
        st.warning(f"خطا در اجرای Prophet: {e}\nاز Moving Average برای پیش‌بینی استفاده می‌شود.")
        forecast_vals = np.array(moving_avg_forecast(df["Close"], forecast_days))
        forecast_df = pd.DataFrame({"Date":[d.date() for d in future_dates],"yhat":np.round(forecast_vals,2)})
else:
    forecast_vals = np.array(moving_avg_forecast(df["Close"], forecast_days))
    forecast_df = pd.DataFrame({"Date":[d.date() for d in future_dates],"yhat":np.round(forecast_vals,2)})

# ensure 1-d
forecast_vals = np.asarray(forecast_vals).reshape(-1,)

# show forecast table
st.markdown("### جدول پیش‌بینی")
st.dataframe(forecast_df.reset_index(drop=True), use_container_width=True)
st.download_button("دانلود پیش‌بینی (CSV)", forecast_df.to_csv(index=False).encode("utf-8"), file_name=f"{symbol}_forecast.csv")

# -----------------------
# Forecast chart
# -----------------------
fig_f = go.Figure()
fig_f.add_trace(go.Scatter(x=df["Date"].tolist(), y=df["Close"].tolist(), mode="lines", name="Actual", line=dict(color="#9AA0A6")))
fig_f.add_trace(go.Scatter(x=[d for d in future_dates], y=forecast_vals.tolist(), mode="lines+markers", name="Forecast", line=dict(color="#FFA500", dash="dot")))
# uncertainty band if prophet used
if used_prophet and "yhat_lower" in forecast_df.columns and "yhat_upper" in forecast_df.columns:
    lower = forecast_df["yhat_lower"].values
    upper = forecast_df["yhat_upper"].values
    xs = list(pd.to_datetime(forecast_df["Date"])) + list(pd.to_datetime(forecast_df["Date"])[::-1])
    ys = list(upper) + list(lower[::-1])
    fig_f.add_trace(go.Scatter(x=xs, y=ys, fill="toself", fillcolor="rgba(255,165,0,0.12)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=False))
fig_f.update_layout(template="plotly_dark", height=460, margin=dict(t=30,b=10))
st.plotly_chart(fig_f, use_container_width=True)

# -----------------------
# Compute and show buy/sell/hold signal
# -----------------------
next_pred = float(forecast_vals[0]) if len(forecast_vals)>0 else None
signal_text, signal_reason = compute_signals(df, forecast_next=next_pred)

st.markdown("## سیگنال معاملاتی")
signal_color = "green" if "خرید" in signal_text or "BUY" in signal_text else ("red" if "فروش" in signal_text or "SELL" in signal_text else "goldenrod")
st.markdown(f"<div style='padding:16px; border-radius:8px; background-color:#111218;'><h2 style='color:{signal_color}; margin:0;'>{signal_text}</h2><p style='color:#bdbdbd; margin:4px 0 0 0;'>{signal_reason}</p></div>", unsafe_allow_html=True)

# -----------------------
# Optional: show RSI/MACD last values
# -----------------------
if HAS_TA:
    try:
        rsi = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        macd = ta.trend.MACD(df["Close"])
        macd_diff = macd.macd_diff()
        st.markdown("### Indicators (Latest)")
        st.write({
            "RSI (latest)": f"{float(rsi.iloc[-1]):.2f}",
            "MACD diff (latest)": f"{float(macd_diff.iloc[-1]):.6f}"
        })
    except Exception:
        pass
else:
    st.info("برای نمایش RSI/MACD بستهٔ 'ta' نصب نشده — برای نصب: pip install ta")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.caption("AureumAI · Demo — اطلاعات صرفاً آموزشی است و این یک توصیه سرمایه‌گذاری نیست.")
