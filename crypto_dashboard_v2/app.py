import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="داشبورد تحلیل ارز دیجیتال", layout="wide")

# عنوان برنامه
st.title("📊 داشبورد تحلیل ارزهای دیجیتال")

# انتخاب ارز دیجیتال
crypto_symbol = st.selectbox(
    "ارز دیجیتال مورد نظر را انتخاب کنید:",
    ["BTC-USD", "ETH-USD", "ADA-USD", "XLM-USD", "BNB-USD", "SOL-USD"]
)
crypto_name = crypto_symbol.split("-")[0]

# بازه تاریخی
days = st.slider("تعداد روزهای گذشته برای نمایش:", 7, 365, 90)

# دانلود داده‌ها
end_date = datetime.now()
start_date = end_date - timedelta(days=days)
data = yf.download(crypto_symbol, start=start_date, end=end_date)

st.subheader(f"داده‌های قیمتی {crypto_name}")
st.dataframe(data.tail())

# محاسبه تغییرات 24 ساعته
if len(data) >= 2:
    last = data["Close"].iloc[-1]
    prev = data["Close"].iloc[-2]
    change24 = ((last - prev) / prev * 100) if prev != 0 else 0.0
    st.metric(label=f"تغییرات 24 ساعته ({crypto_name})", value=f"{change24:.2f}%", delta=f"{change24:.2f}%")
else:
    st.warning("داده کافی برای محاسبه تغییرات 24 ساعته وجود ندارد.")

# رسم نمودار فقط در صورت وجود داده معتبر
if "Close" in data.columns and not bool(data["Close"].isna().all()):
    fig = px.line(
        data.reset_index(),
        x="Date",
        y="Close",
        title=f"روند قیمت {crypto_name}",
        labels={"Close": "قیمت پایانی", "Date": "تاریخ"}
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("داده‌ای برای رسم نمودار پیدا نشد.")

# حجم معاملات
if "Volume" in data.columns:
    st.subheader("حجم معاملات")
    st.bar_chart(data["Volume"])
else:
    st.warning("داده‌ای برای حجم معاملات موجود نیست.")
