import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

# عنوان اپ
st.set_page_config(page_title="داشبورد تحلیل ارز دیجیتال", layout="wide")

st.title("📊 داشبورد تحلیل ارز دیجیتال")
st.markdown("### داده‌ها و تحلیل بازار ارزهای دیجیتال (BTC, ADA, XLM و ...)")

# انتخاب ارز و بازه زمانی
coins = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Stellar (XLM-USD)": "XLM-USD",
    "Ethereum (ETH-USD)": "ETH-USD"
}
coin_name = st.selectbox("ارز مورد نظر:", list(coins.keys()))
symbol = coins[coin_name]

period = st.selectbox("بازه زمانی:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
interval = st.selectbox("فاصله داده‌ها:", ["1d", "1wk", "1mo"])

# دانلود داده‌ها
@st.cache_data
def load_data(symbol, period, interval):
    data = yf.download(symbol, period=period, interval=interval)
    if "Close" not in data.columns:
        return pd.DataFrame()
    data = data.dropna(subset=["Close"])
    return data

data = load_data(symbol, period, interval)

if data.empty:
    st.error("❌ داده‌ای برای این ارز و بازه زمانی یافت نشد.")
    st.stop()

# نمایش قیمت فعلی و تغییرات
last_price = data["Close"].iloc[-1]
prev_price = data["Close"].iloc[-2] if len(data) > 1 else last_price
change_percent = ((last_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("💰 قیمت فعلی", f"${last_price:,.4f}")
col2.metric("📈 تغییر 24 ساعته", f"{change_percent:.2f} %")
col3.metric("📅 آخرین تاریخ", data.index[-1].strftime("%Y-%m-%d"))

# نمودار قیمت
fig = px.line(
    data.reset_index(),
    x="Date",
    y="Close",
    title=f"نمودار قیمت {coin_name}",
    labels={"Close": "قیمت پایانی", "Date": "تاریخ"}
)
st.plotly_chart(fig, use_container_width=True)

# نمایش داده‌های خام
st.markdown("### 📋 داده‌های تاریخی")
st.dataframe(data.tail(20))

# دانلود فایل CSV
csv = data.to_csv().encode("utf-8")
st.download_button(
    label="📥 دانلود داده‌ها (CSV)",
    data=csv,
    file_name=f"{symbol}_data.csv",
    mime="text/csv",
)
