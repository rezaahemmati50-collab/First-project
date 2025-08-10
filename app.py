# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import io
from prophet import Prophet
import ta
from datetime import datetime

st.set_page_config(page_title="Crypto & Stock Forecast", layout="centered")

# -----------------------
# Utilities
# -----------------------
def safe_extract_close(df, symbol=None):
    """
    Try to extract a 1-D close price Series from a yfinance-returned DataFrame.
    Handles normal, MultiIndex (Close, ticker) and some edge cases.
    Returns (series, debug_msg) - series may be None if extraction failed.
    """
    debug = ""
    if df is None or df.empty:
        return None, "no data returned"
    # if columns are MultiIndex (levels), try to find 'Close' level
    if isinstance(df.columns, pd.MultiIndex):
        # try typical layout: ('Close', 'SYMBOL') or ('SYMBOL', 'Close')
        if 'Close' in df.columns.get_level_values(0):
            # select first available second level under Close
            try:
                close_cols = [c for c in df.columns if c[0] == 'Close']
                # prefer the column that matches symbol if given
                if symbol:
                    for c in close_cols:
                        if symbol in str(c[1]):
                            s = df[c].copy()
                            s.index = pd.to_datetime(s.index)
                            s.name = 'Close'
                            return s, ""
                # fallback: take first Close column
                s = df[close_cols[0]].copy()
                s.index = pd.to_datetime(s.index)
                s.name = 'Close'
                return s, ""
            except Exception as e:
                debug = f"MultiIndex present but couldn't extract Close: {e}"
                return None, debug
        # maybe 'Close' is in second level
        if 'Close' in df.columns.get_level_values(1):
            try:
                close_cols = [c for c in df.columns if c[1] == 'Close']
                s = df[close_cols[0]].copy()
                s.index = pd.to_datetime(s.index)
                s.name = 'Close'
                return s, ""
            except Exception as e:
                debug = f"MultiIndex present but couldn't extract Close (level 1): {e}"
                return None, debug
        # otherwise fail
        return None, "MultiIndex present but no Close level found."
    else:
        # single index columns
        if 'Close' in df.columns:
            s = df['Close'].copy()
            s.index = pd.to_datetime(s.index)
            s.name = 'Close'
            return s, ""
        # sometimes yfinance returns columns lowercase or different naming
        for candidate in ['close', 'Adj Close', 'AdjClose', 'adjclose']:
            if candidate in df.columns:
                s = df[candidate].copy()
                s.index = pd.to_datetime(s.index)
                s.name = 'Close'
                return s, f"used {candidate} as Close"
        return None, "No 'Close' column found."

def compute_indicators(price_series):
    """
    price_series: pd.Series indexed by datetime
    Returns dict with MA20, MA50, RSI, MACD_diff (all aligned)
    """
    res = {}
    s = price_series.ffill().dropna()
    if s.empty:
        return res
    df = pd.DataFrame({'close': s})
    df['MA20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['MA50'] = df['close'].rolling(window=50, min_periods=1).mean()
    try:
        rsi = ta.momentum.RSIIndicator(df['close']).rsi()
        macd_diff = ta.trend.MACD(df['close']).macd_diff()
    except Exception:
        # fallback simple NaNs
        rsi = pd.Series(np.nan, index=df.index)
        macd_diff = pd.Series(np.nan, index=df.index)
    res['MA20'] = df['MA20']
    res['MA50'] = df['MA50']
    res['RSI'] = rsi
    res['MACD_diff'] = macd_diff
    return res

def generate_signal_from_indicators(indicat):
    try:
        if not indicat:
            return "⚠️ اندیکاتور موجود نیست"
        rsi = indicat['RSI']
        macd = indicat['MACD_diff']
        if rsi.empty or macd.empty:
            return "⚠️ اندیکاتور ناکافی"
        last_rsi = rsi.dropna().iloc[-1] if not rsi.dropna().empty else np.nan
        last_macd = macd.dropna().iloc[-1] if not macd.dropna().empty else np.nan
        if np.isnan(last_rsi) or np.isnan(last_macd):
            return "⚠️ اندیکاتور ناکافی"
        if last_rsi < 30 and last_macd > 0:
            return "🔵 سیگنال خرید (Buy)"
        elif last_rsi > 70 and last_macd < 0:
            return "🔴 سیگنال فروش (Sell)"
        else:
            return "🟡 نگه‌داری (Hold)"
    except Exception as e:
        return f"⚠️ خطا در تولید سیگنال: {e}"

def prepare_prophet_df_from_series(series):
    """
    series: pandas Series indexed by datetime
    returns DataFrame with columns ds (datetime) and y (float)
    """
    df = series.dropna().to_frame(name='y').reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    # ensure y is numeric 1d
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['y'])
    return df

def predict_prophet(series, days=3):
    """
    series: pd.Series indexed by datetime
    returns forecast dataframe (ds, yhat, yhat_lower, yhat_upper)
    """
    df = prepare_prophet_df_from_series(series)
    if df.shape[0] < 2:
        raise ValueError("Dataframe has less than 2 non-NaN rows for Prophet.")
    m = Prophet(daily_seasonality=True)
    # silence logging
    m.fit(df)
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)

# -----------------------
# UI
# -----------------------
st.title("📈 Crypto & Stock Forecast — پیش‌بینی قیمت")

st.markdown("""
نسخه پیشنهادی: Prophet + تحلیل تکنیکال ساده (RSI, MACD, MA20/50).
- می‌توانید نماد را از فهرست انتخاب کنید یا دستی وارد کنید.
- امکان آپلود CSV نمونه (فرمت: ds,y) برای پیش‌بینی آفلاین وجود دارد.
""")

# quick presets
preset_symbols = [
    "BTC-USD", "ETH-USD", "ADA-USD", "XLM-USD", "SOL-USD",
    "LTC-USD", "XRP-USD", "DOGE-USD", "DOT-USD"
]

symbol_choice = st.selectbox("انتخاب از فهرست نمادها (یا از ورودی دستی استفاده کنید):", ["(use manual input)"] + preset_symbols)
manual_symbol = st.text_input("وارد کردن نماد دستی (مثال: BTC-USD یا AAPL):", value="BTC-USD" if symbol_choice == "(use manual input)" else symbol_choice)
symbol = manual_symbol.strip().upper()

# history and interval
history_period = st.selectbox("بازهٔ تاریخی:", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
interval = st.selectbox("فاصله زمانی:", ["1d", "1h", "1wk"], index=0)

# forecast horizon
horizon = st.radio("Days to forecast:", (3, 7, 30), index=0)

# CSV upload
st.markdown("**یا** فایل CSV آپلود کنید (فرمت: `ds,y` برای دادهٔ تاریخ/قیمت).")
uploaded = st.file_uploader("Upload CSV (optional)", type=['csv'])
user_series = None
if uploaded is not None:
    try:
        df_uploaded = pd.read_csv(uploaded)
        # accept ds,y or Date,Close
        if set(['ds','y']).issubset(df_uploaded.columns):
            df_uploaded['ds'] = pd.to_datetime(df_uploaded['ds'])
            s = pd.Series(df_uploaded['y'].values, index=df_uploaded['ds'])
            user_series = s.sort_index()
            st.success("فایل بارگذاری شد (ds,y).")
        elif set(['Date','Close']).issubset(df_uploaded.columns):
            df_uploaded['Date'] = pd.to_datetime(df_uploaded['Date'])
            s = pd.Series(df_uploaded['Close'].values, index=df_uploaded['Date'])
            user_series = s.sort_index()
            st.success("فایل بارگذاری شد (Date,Close).")
        else:
            st.error("فرمت CSV شناخته نشد — از ستون‌های ds,y یا Date,Close استفاده کنید.")
    except Exception as e:
        st.error(f"خطا در خواندن فایل: {e}")

# fetch button
col1, col2 = st.columns([1, 1])
with col1:
    btn_fetch = st.button("دریافت و تحلیل دیتا")
with col2:
    auto_refresh = st.checkbox("Auto-refresh every 60s (works while page active)")

if btn_fetch:
    st.info(f"در حال دریافت داده برای {symbol} ...")
    try:
        df_raw = yf.download(symbol, period=history_period, interval=interval, progress=False, threads=False)
    except Exception as e:
        st.error(f"خطا در دریافت داده: {e}")
        df_raw = None

    close_series, debug = safe_extract_close(df_raw, symbol)
    if close_series is None and user_series is None:
        st.error(f"⚠️ Could not extract Close for {symbol}. debug: {debug}")
    else:
        used_series = user_series if user_series is not None else close_series
        # show last price and basic metrics
        latest_price = None
        try:
            latest_price = float(used_series.dropna().iloc[-1])
        except Exception:
            latest_price = None

        if latest_price is not None:
            st.metric(label=f"{symbol} Latest price", value=f"${latest_price:,.2f}")
        # indicators
        indic = compute_indicators(used_series)
        signal = generate_signal_from_indicators(indic)

        # display chart
        st.subheader("Price chart & indicators")
        plot_df = pd.DataFrame({'Close': used_series})
        # add MAs if exist
        if 'MA20' in indic:
            plot_df['MA20'] = indic['MA20']
        if 'MA50' in indic:
            plot_df['MA50'] = indic['MA50']
        st.line_chart(plot_df)

        st.subheader("Technical signal")
        st.markdown(f"### {signal}")

        # Prophet forecast
        st.subheader(f"Predict next {horizon} days (Prophet)")
        try:
            forecast = predict_prophet(used_series, days=horizon)
            # format for display
            disp = forecast.copy()
            disp['ds'] = disp['ds'].dt.date
            disp[['yhat','yhat_lower','yhat_upper']] = disp[['yhat','yhat_lower','yhat_upper']].round(6)
            st.table(disp.reset_index(drop=True))
        except Exception as e:
            st.error(f"⚠️ خطا در پیش‌بینی: {e}")

        # show some debug / source
        st.markdown("---")
        st.caption(f"Source: yfinance. debug: {debug}")

# Auto refresh (simple)
if auto_refresh:
    st.experimental_rerun()

# footer / help
st.markdown("---")
st.markdown("این ابزار صرفاً آموزشی است و مشاورهٔ سرمایه‌گذاری نیست. قبل از هر تصمیم مالی تحقیق کنید.") 
