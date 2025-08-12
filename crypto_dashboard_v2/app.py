import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# عنوان برنامه
st.title("📊 داشبورد قیمت ارزهای دیجیتال")

# انتخاب ارز دیجیتال
cryptos = {
    "بیت‌کوین (BTC)": "BTC-USD",
    "اتریوم (ETH)": "ETH-USD",
    "کاردانو (ADA)": "ADA-USD",
    "ریپل (XRP)": "XRP-USD",
    "استلار (XLM)": "XLM-USD"
}
crypto_name = st.selectbox("یک ارز انتخاب کنید:", list(cryptos.keys()))
symbol = cryptos[crypto_name]

# بازه زمانی
period = st.selectbox("بازه زمانی:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"], index=5)

# دریافت داده
data = yf.download(symbol, period=period)

if data.empty:
    st.error("❌ داده‌ای برای این ارز پیدا نشد!")
else:
    # محاسبه تغییرات 24 ساعته
    if len(data) >= 2:
        last = float(data["Close"].iloc[-1])
        prev = float(data["Close"].iloc[-2])
        change24 = (last - prev) / prev * 100 if prev != 0 else 0.0
    else:
        last = prev = change24 = 0.0

    st.metric(label=f"📈 قیمت پایانی {crypto_name}", value=f"${last:,.2f}", delta=f"{change24:.2f}%")

    # نمایش نمودار
    fig = px.line(
        data,
        x=data.index,
        y="Close",
        title=f"روند قیمت {crypto_name}",
        labels={"Close": "قیمت پایانی", "index": "تاریخ"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # نمایش جدول
    st.subheader("📋 داده‌های خام")
    st.dataframe(data.tail(10))
