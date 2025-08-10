import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date

# تنظیمات اولیه
st.title("📈 پیش‌بینی قیمت ارز دیجیتال / سهام")
st.write("این برنامه با استفاده از داده‌های Yahoo Finance و مدل Prophet قیمت آینده را پیش‌بینی می‌کند.")

# ورودی‌ها
ticker = st.text_input("نماد (مثال: BTC-USD یا AAPL)", value="BTC-USD")
years = st.slider("مدت پیش‌بینی (سال)", 1, 4)
period = years * 365

# دانلود داده‌ها
@st.cache_data
def load_data(ticker):
    try:
        df = yf.download(ticker, period="max")
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"خطا در دریافت داده‌ها: {e}")
        return None

if st.button("پیش‌بینی کن"):
    df = load_data(ticker)

    if df is None or "Close" not in df.columns:
        st.error("❌ ستون Close پیدا نشد. نماد را بررسی کنید.")
    else:
        # تبدیل ستون Close به عددی
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])

        st.subheader("📊 داده‌های اخیر")
        st.dataframe(df.tail())

        # آماده‌سازی داده برای Prophet
        df_train = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

        # مدل Prophet
        m = Prophet(daily_seasonality=True)
        m.fit(df_train)

        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        st.subheader("📈 پیش‌بینی قیمت")
        fig1 = plot_plotly(m, forecast)
        st.plotly_chart(fig1)

        st.subheader("🔍 جزئیات پیش‌بینی")
        st.write(forecast.tail())
