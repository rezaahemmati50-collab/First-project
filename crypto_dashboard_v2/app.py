import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px

st.set_page_config(page_title="داشبورد ارز دیجیتال", layout="wide")

st.title("📊 داشبورد تحلیل ارز دیجیتال")

# انتخاب ارز
cryptos = {
    "BTC-USD": "بیت کوین",
    "ETH-USD": "اتریوم",
    "ADA-USD": "کاردانو",
    "XRP-USD": "ریپل",
    "XLM-USD": "استلار"
}

crypto_symbol = st.selectbox("ارز مورد نظر را انتخاب کنید:", list(cryptos.keys()), format_func=lambda x: cryptos[x])

# بازه زمانی
period = st.selectbox("بازه زمانی:", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "max"], index=5)

# دریافت داده‌ها
data = yf.download(crypto_symbol, period=period, interval="1d")

if not data.empty:
    # محاسبه تغییرات ۲۴ ساعته (اصلاح شده)
    if len(data) >= 2:
        last = float(data["Close"].iloc[-1])
        prev = float(data["Close"].iloc[-2])
        change24 = ((last - prev) / prev * 100) if prev != 0 else 0.0
        st.metric(label=f"تغییرات 24 ساعته ({cryptos[crypto_symbol]})", value=f"{change24:.2f}%", delta=f"{change24:.2f}%")
    else:
        st.warning("داده کافی برای محاسبه تغییرات 24 ساعته وجود ندارد.")

    # نمایش نمودار
    fig = px.line(
        data,
        x=data.index,
        y="Close",
        title=f"روند قیمت {cryptos[crypto_symbol]}",
        labels={"Close": "قیمت پایانی", "index": "تاریخ"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # جدول داده‌ها
    st.subheader("داده‌های خام")
    st.dataframe(data)
else:
    st.error("هیچ داده‌ای برای این ارز و بازه زمانی پیدا نشد.")
