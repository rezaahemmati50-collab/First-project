import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
from datetime import datetime, timedelta

# عنوان برنامه
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("📊 Crypto Dashboard")

# لیست ارزها
cryptos = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "Ripple (XRP)": "XRP-USD",
    "Stellar (XLM)": "XLM-USD"
}

# انتخاب ارز
crypto_name = st.selectbox("ارز دیجیتال را انتخاب کنید:", list(cryptos.keys()))
crypto_symbol = cryptos[crypto_name]

# انتخاب بازه زمانی
periods = {
    "7 روز گذشته": 7,
    "1 ماه گذشته": 30,
    "3 ماه گذشته": 90,
    "6 ماه گذشته": 180,
    "1 سال گذشته": 365
}
period_name = st.selectbox("بازه زمانی:", list(periods.keys()))
days = periods[period_name]

# دریافت داده‌ها
end_date = datetime.today()
start_date = end_date - timedelta(days=days)
data = yf.download(crypto_symbol, start=start_date, end=end_date)

# نمایش جدول داده‌ها
st.subheader(f"داده‌های {crypto_name}")
st.dataframe(data.tail(10))

# نمایش نمودار
fig = px.line(data, x=data.index, y="Close", title=f"قیمت پایانی {crypto_name}")
st.plotly_chart(fig, use_container_width=True)

# میانگین متحرک ساده (SMA)
data["SMA"] = data["Close"].rolling(window=5).mean()
st.subheader("📈 نمودار همراه با SMA (میانگین متحرک)")
fig_sma = px.line(data, x=data.index, y=["Close", "SMA"], title="SMA Trend")
st.plotly_chart(fig_sma, use_container_width=True)

# توضیح پایانی
st.markdown("این داشبورد با استفاده از **Streamlit** و داده‌های زنده **Yahoo Finance** ساخته شده است.")
