import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
from prophet import Prophet

st.set_page_config(page_title="تحلیل بازار ارز دیجیتال", layout="centered")

# دریافت داده از یاهو فایننس
def get_data(symbol):
    return yf.download(symbol, period="3mo", interval="1d")

# تحلیل تکنیکال با RSI و MACD
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
            return "🔵 سیگنال خرید (Buy)"
        elif last_rsi > 70 and last_macd < 0:
            return "🔴 سیگنال فروش (Sell)"
        else:
            return "🟡 نگه‌داری (Hold)"
    except Exception as e:
        return f"⚠️ خطا در محاسبه اندیکاتورها: {e}"

# پیش‌بینی قیمت با Prophet
def predict_with_prophet(data, days=3):
    df = data[['Close']].copy().reset_index()

    if 'Date' in df.columns:
        df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    elif 'index' in df.columns:
        df.rename(columns={'index': 'ds', 'Close': 'y'}, inplace=True)
    else:
        df.rename(columns={df.columns[0]: 'ds', 'Close': 'y'}, inplace=True)

    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = df['y'].astype(float)  # اصلاح شده: فقط تبدیل نوع کافی است

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    predicted = forecast[['ds', 'yhat']].tail(days)
    return predicted

# رابط کاربری
st.title("📊 تحلیل و پیش‌بینی بازار ارز دیجیتال")

assets = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "Stellar (XLM)": "XLM-USD",
    "Solana (SOL)": "SOL-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Ripple (XRP)": "XRP-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Polkadot (DOT)": "DOT-USD"
}

asset_name = st.selectbox("🪙 انتخاب ارز دیجیتال:", list(assets.keys()))
symbol = assets[asset_name]

with st.spinner("⏳ در حال دریافت داده‌ها..."):
    data = get_data(symbol)

if data.empty:
    st.error("⚠️ داده‌ای برای این ارز پیدا نشد.")
    st.stop()

st.subheader("📈 نمودار قیمت")
st.line_chart(data['Close'])

st.subheader("📌 سیگنال تحلیل تکنیکال")
signal = generate_signal(data)
st.markdown(f"### {signal}")

st.subheader("🤖 پیش‌بینی قیمت با Prophet (۳ روز آینده)")
try:
    predicted_df = predict_with_prophet(data, days=3)
    predicted_df['yhat'] = predicted_df['yhat'].round(2)
    predicted_df['ds'] = predicted_df['ds'].dt.date
    predicted_df.columns = ['تاریخ', 'قیمت پیش‌بینی‌شده (USD)']
    st.table(predicted_df)
except Exception as e:
    st.error(f"⚠️ خطا در پیش‌بینی قیمت: {e}")
