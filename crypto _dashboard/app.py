import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

# عنوان برنامه
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("📊 داشبورد تحلیل ارز دیجیتال")

# لیست ارزها
cryptos = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Cardano": "ADA-USD",
    "Ripple": "XRP-USD",
    "Stellar": "XLM-USD"
}

# انتخاب ارز
crypto_name = st.selectbox("یک ارز انتخاب کنید:", list(cryptos.keys()))
symbol = cryptos[crypto_name]

# انتخاب بازه زمانی
period_options = {
    "1 روز": "1d",
    "5 روز": "5d",
    "1 ماه": "1mo",
    "6 ماه": "6mo",
    "1 سال": "1y",
    "5 سال": "5y",
    "کل داده‌ها": "max"
}
period = st.selectbox("بازه زمانی:", list(period_options.keys()))
period_value = period_options[period]

# دریافت داده‌ها
data = yf.download(symbol, period=period_value)

# بررسی خالی نبودن دیتا
if data.empty:
    st.error("⛔ داده‌ای برای این ارز پیدا نشد. لطفاً بازه زمانی دیگری انتخاب کنید.")
    st.stop()

# نمودار قیمت پایانی
fig = px.line(data, x=data.index, y="Close", title=f"قیمت پایانی {crypto_name}")
st.plotly_chart(fig, use_container_width=True)

# محاسبه تغییرات ۲۴ ساعته
if len(data) >= 2:
    last = data["Close"].iloc[-1]
    prev = data["Close"].iloc[-2]
    change24 = (last - prev) / prev * 100 if prev != 0 else 0.0
    st.metric(label="درصد تغییرات ۲۴ ساعت اخیر", value=f"{change24:.2f}%")
else:
    st.warning("📉 داده کافی برای محاسبه تغییرات ۲۴ ساعته وجود ندارد.")

# نمایش جدول داده‌ها
st.dataframe(data.tail(10))
