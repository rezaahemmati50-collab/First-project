import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta

# عنوان اپ
st.title("📈 داشبورد تحلیل و پیش‌بینی ارز دیجیتال")

# لیست ارزها
coins = {
    "Bitcoin (BTC)": "BTC-USD",
    "Cardano (ADA)": "ADA-USD",
    "Stellar (XLM)": "XLM-USD",
    "Ethereum (ETH)": "ETH-USD"
}

coin_name = st.selectbox("ارز مورد نظر را انتخاب کنید:", list(coins.keys()))
symbol = coins[coin_name]

# انتخاب بازه زمانی
time_ranges = {
    "1 روز": ("1d", "5m"),
    "5 روز": ("5d", "30m"),
    "1 هفته": ("7d", "1h"),
    "2 هفته": ("14d", "2h")
}
selected_range = st.selectbox("بازه زمانی:", list(time_ranges.keys()))
period, interval = time_ranges[selected_range]

# دانلود داده
data = yf.download(symbol, period=period, interval=interval)
if data.empty:
    st.error("❌ داده‌ای برای این بازه زمانی موجود نیست.")
    st.stop()

# محاسبه میانگین‌های متحرک
data["MA20"] = data["Close"].rolling(window=20).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()

# آخرین قیمت و سیگنال خرید/فروش
latest_price = data["Close"].iloc[-1]
ma20 = data["MA20"].dropna().iloc[-1] if not data["MA20"].dropna().empty else None
ma50 = data["MA50"].dropna().iloc[-1] if not data["MA50"].dropna().empty else None

signal = "⚪ صبر کنید"
if ma20 and ma50:
    if latest_price > ma20 > ma50:
        signal = "🟢 خرید"
    elif latest_price < ma20 < ma50:
        signal = "🔴 فروش"

# نمایش جدول و نمودار
st.metric(label="آخرین قیمت", value=f"${latest_price:,.2f}")
st.metric(label="سیگنال", value=signal)
st.line_chart(data[["Close", "MA20", "MA50"]])

# پیش‌بینی قیمت با Prophet
df_prophet = data.reset_index()[["Date", "Close"]]
df_prophet.columns = ["ds", "y"]

model = Prophet(daily_seasonality=True)
model.fit(df_prophet)

future = model.make_future_dataframe(periods=7)  # فقط یکبار اجرا
forecast = model.predict(future)

# نمایش پیش‌بینی ۳ و ۷ روز آینده
pred_3d = forecast.iloc[-4]["yhat"]
pred_7d = forecast.iloc[-1]["yhat"]

st.subheader("پیش‌بینی قیمت")
st.write(f"📅 ۳ روز آینده: **${pred_3d:,.2f}**")
st.write(f"📅 ۷ روز آینده: **${pred_7d:,.2f}**")
st.line_chart(forecast.set_index("ds")[["yhat"]])
