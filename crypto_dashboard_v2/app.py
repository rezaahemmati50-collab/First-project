import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# عنوان داشبورد
st.title("📊 داشبورد قیمت ارزهای دیجیتال")

# انتخاب ارز
symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "DOGE-USD", "SOL-USD"]
symbol = st.selectbox("ارز مورد نظر را انتخاب کنید:", symbols)

# دانلود داده‌ها
try:
    data = yf.download(symbol, period="7d", interval="1h")

    if data.empty or "Close" not in data.columns:
        st.error(f"داده‌ای برای {symbol} پیدا نشد.")
    else:
        # حذف سطرهای خالی
        data = data.dropna(subset=["Close"])

        # محاسبه تغییر ۲۴ ساعت اخیر
        last = data["Close"].iloc[-1]
        prev_time = data.index[-1] - timedelta(hours=24)
        prev = data.loc[data.index >= prev_time, "Close"].iloc[0]

        change24 = ((last - prev) / prev * 100) if prev != 0 else 0.0

        # نمایش قیمت و تغییر
        st.metric(label=f"قیمت فعلی {symbol}", value=f"${last:,.4f}", delta=f"{change24:.2f}%")

        # رسم نمودار
        fig = px.line(
            data.reset_index(),
            x="Datetime" if "Datetime" in data.reset_index().columns else "Date",
            y="Close",
            title=f"روند قیمت {symbol}",
            labels={"Close": "قیمت پایانی", "Datetime": "تاریخ", "Date": "تاریخ"}
        )
        st.plotly_chart(fig, use_container_width=True)

        # نمایش جدول
        st.dataframe(data.tail(10))

except Exception as e:
    st.error(f"مشکلی پیش آمد: {e}")
