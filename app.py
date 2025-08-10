import streamlit as st
import pandas as pd
from prophet import Prophet
import yfinance as yf

st.title("📈 پیش‌بینی قیمت ارز دیجیتال")

# انتخاب ارز
symbol = st.text_input("نماد ارز دیجیتال را وارد کنید (مثلاً BTC-USD):", "BTC-USD")

# دریافت داده
if st.button("📥 دریافت و پیش‌بینی"):
    try:
        df = yf.download(symbol, period="1y")

        if df.empty:
            st.error("❌ داده‌ای برای این نماد پیدا نشد.")
            st.stop()

        # بررسی وجود ستون Close
        if 'Close' not in df.columns:
            st.error("❌ ستون Close در داده‌ها پیدا نشد.")
            st.stop()

        close_col = df['Close']

        # بررسی نوع داده
        if not isinstance(close_col, (pd.Series, list, tuple)):
            st.error("❌ ستون Close فرمت نامناسب دارد.")
            st.stop()

        # بررسی خالی بودن
        if pd.Series(close_col).empty:
            st.error("❌ ستون Close خالی است.")
            st.stop()

        # آماده‌سازی داده برای Prophet
        df_train = pd.DataFrame({
            "ds": df.index,
            "y": pd.to_numeric(close_col, errors='coerce')
        }).dropna()

        if df_train.empty:
            st.error("❌ داده مناسب برای پیش‌بینی وجود ندارد.")
            st.stop()

        # مدل Prophet
        model = Prophet()
        model.fit(df_train)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        st.subheader("📊 نمودار پیش‌بینی")
        st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])

    except Exception as e:
        st.error(f"🚨 خطا در پردازش داده: {e}")
