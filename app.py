import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.title("📈 پیش‌بینی قیمت ارز دیجیتال / سهام")

# ورودی تیکرها
tickers = st.text_input("🔍 نماد یا نمادها را وارد کنید (با کاما جدا کنید)", "BTC-USD")

period = st.slider("مدت پیش‌بینی (سال)", 1, 4, 1)

if st.button("📊 پیش‌بینی کن"):
    tickers_list = [t.strip() for t in tickers.split(",")]

    for ticker in tickers_list:
        st.subheader(f"🔹 {ticker}")

        try:
            # دانلود دیتا فقط برای همین تیکر
            df = yf.download(ticker, period="5y")
            if df.empty:
                st.error(f"❌ داده‌ای برای {ticker} پیدا نشد.")
                continue

            # انتخاب ستون Close به صورت Series
            close_prices = df["Close"]
            close_prices = pd.to_numeric(close_prices, errors="coerce")

            # آماده‌سازی دیتا برای Prophet
            df_train = pd.DataFrame({
                "ds": close_prices.index,
                "y": close_prices.values
            }).dropna()

            # مدل Prophet
            m = Prophet()
            m.fit(df_train)

            future = m.make_future_dataframe(periods=period * 365)
            forecast = m.predict(future)

            # نمودار
            fig1 = plot_plotly(m, forecast)
            st.plotly_chart(fig1)

            # داده‌های پیش‌بینی
            st.write(forecast.tail())

        except Exception as e:
            st.error(f"⚠️ خطا در پردازش {ticker}: {e}")
