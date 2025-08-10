import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import datetime

# لیست ارزها
coins = {
    "BTC": "Bitcoin",
    "ETH": "Ethereum",
    "ADA": "Cardano",
    "XLM": "Stellar"
}

# انتخاب ارز
symbol = st.selectbox("یک ارز دیجیتال انتخاب کنید:", list(coins.keys()))

# انتخاب بازه زمانی
period_option = st.selectbox("بازه زمانی:", ["1d", "5d", "1wk", "2wk", "1mo", "3mo", "6mo", "1y"])

# تعیین interval مناسب
if period_option in ["1d", "5d"]:
    interval = "5m"
else:
    interval = "1d"

# دانلود داده
data = yf.download(f"{symbol}-USD", period=period_option, interval=interval)

# بررسی خالی نبودن داده
if data.empty or data["Close"].dropna().empty:
    st.warning("⚠️ هیچ داده‌ای برای این بازه یافت نشد. لطفاً بازه یا ارز دیگری انتخاب کنید.")
else:
    # آخرین قیمت
    latest_price = data["Close"].dropna().iloc[-1]
    st.metric(label="💰 آخرین قیمت", value=f"${latest_price:,.2f}")

    # محاسبه میانگین‌های متحرک
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()

    # رسم نمودار
    st.line_chart(data[["Close", "MA20", "MA50"]])

    # آماده‌سازی دیتا برای Prophet
    df = data.reset_index()[["Date", "Close"]]
    df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

    if len(df.dropna()) > 2:
        m = Prophet()
        m.fit(df.dropna())

        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)

        st.subheader("پیش‌بینی ۳۰ روز آینده")
        st.line_chart(forecast[["ds", "yhat"]].set_index("ds"))
    else:
        st.info("📉 داده کافی برای پیش‌بینی وجود ندارد.")
