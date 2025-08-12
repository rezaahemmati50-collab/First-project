import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Dashboard", layout="wide")

st.title("📊 داشبورد تحلیل ارز دیجیتال")

cryptos = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "Ripple (XRP)": "XRP-USD",
    "Solana (SOL)": "SOL-USD"
}

crypto_name = st.selectbox("یک ارز انتخاب کنید:", list(cryptos.keys()))
symbol = cryptos[crypto_name]

period = st.selectbox("بازه زمانی:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=5)

data = yf.download(symbol, period=period, interval="1d")

if not data.empty:
    data = data.dropna(subset=["Close"])

    last = data["Close"].iloc[-1]
    prev = data["Close"].iloc[-2] if len(data) > 1 else last
    change24 = ((last - prev) / prev * 100) if prev != 0 else 0.0

    col1, col2 = st.columns(2)
    col1.metric(f"قیمت پایانی {crypto_name}", f"${last:,.4f}", f"{change24:.2f}%")
    col2.write(f"آخرین بروزرسانی: {data.index[-1].strftime('%Y-%m-%d')}")

    fig = px.line(data.reset_index(), x="Date", y="Close",
                  title=f"نمودار قیمت {crypto_name}",
                  labels={"Close": "قیمت پایانی", "Date": "تاریخ"})
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(data.tail(10))
else:
    st.warning("⚠️ داده‌ای برای این بازه زمانی موجود نیست.")
