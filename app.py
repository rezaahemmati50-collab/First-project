import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
from prophet import Prophet

# تنظیمات صفحه
st.set_page_config(page_title="تحلیل بازار ارز دیجیتال", page_icon="📈", layout="centered")

# ----------------------------------------
# دریافت داده از Yahoo Finance
# ----------------------------------------
def get_data(symbol):
    data = yf.download(symbol, period="3mo", interval="1d")
    return data

# ----------------------------------------
# تولید سیگنال خرید/فروش با RSI و MACD
# ----------------------------------------
def generate_signal(data):
    if data.empty or 'Close' not in data.columns:
        return "⚠️ داده‌ای برای تحلیل وجود ندارد"

    close = data['Close'].ffill()
    close = pd.Series(close.values.flatten(), index=close.index)

    try:
        rsi = ta.momentum.RSIIndicator(close).rsi()
        macd = ta.trend.MACD(close).macd_diff()

        last_rsi = rsi.iloc[-1]
        last_macd = macd.iloc[-1]

        if last_rsi < 30 and last_macd > 0:
            return "🔵 خرید (Buy)"
        elif last_rsi > 70 and last_macd < 0:
            return "🔴 فروش (Sell)"
        else:
            return "🟡 نگه‌داری (Hold)"
    except Exception as e:
        return f"⚠️ خطا در محاسبه اندیکاتورها: {e}"

# ----------------------------------------
# پیش‌بینی قیمت با مدل Prophet (۳ روز آینده)
# ----------------------------------------
def predict_with_prophet(data, days=3):
    df = data[['Close']].copy()
    df = df.reset_index()

    # شناسایی ستون تاریخ
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    elif 'index' in df.columns:
        df.rename(columns={'index': 'ds', 'Close': 'y'}, inplace=True)

    # اطمینان از فرمت صحیح
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = df['y'].astype(float)

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    predicted = forecast[['ds', 'yhat']].tail(days)
    return predicted

# ----------------------------------------
# رابط کاربری Streamlit
# ----------------------------------------
st.title("📊 تحلیل و پیش‌بینی ارز دیجیتال")

assets = {
    "Bitcoin (BTC)": "BTC-USD",
    "Cardano (ADA)": "ADA-USD",
    "Stellar (XLM)": "XLM-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD"
}

asset_name = st.selectbox("✅ انتخاب ارز:", list(assets.keys()))
symbol = assets[asset_name]

with st.spinner("در حال دریافت داده‌ها..."):
    data = get_data(symbol)

if data.empty:
    st.error("⚠️ داده‌ای برای این ارز پیدا نشد.")
    st.stop()

# نمایش نمودار قیمت
st.subheader("📈 نمودار قیمت بسته شدن")
st.line_chart(data['Close'])

# سیگنال خرید/فروش
st.subheader(f"📌 سیگنال پیشنهادی برای {asset_name}:")
signal = generate_signal(data)
st.markdown(f"### {signal}")

# پیش‌بینی قیمت با Prophet
st.subheader("🤖 پیش‌بینی قیمت با مدل Prophet (۳ روز آینده)")

try:
    predicted_df = predict_with_prophet(data, days=3)
    predicted_df['yhat'] = predicted_df['yhat'].round(2)
    predicted_df['ds'] = predicted_df['ds'].dt.date
    predicted_df.columns = ['تاریخ', 'قیمت پیش‌بینی‌شده (دلار)']

    st.table(predicted_df)
except Exception as e:
    st.error(f"⚠️ خطا در پیش‌بینی قیمت: {e}")
