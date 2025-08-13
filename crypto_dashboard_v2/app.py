import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# عنوان برنامه
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("📊 داشبورد تحلیل قیمت ارز دیجیتال")

# لیست ارزها
coins = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Cardano": "ADA-USD",
    "XRP": "XRP-USD",
    "Litecoin": "LTC-USD"
}

# انتخاب ارز
coin_name = st.selectbox("ارز مورد نظر را انتخاب کنید:", list(coins.keys()))
symbol = coins[coin_name]

# دریافت داده‌ها
@st.cache_data
def load_data(ticker):
    try:
        df = yf.download(ticker, period="1mo", interval="1d")
        if "Close" not in df.columns or df.empty:
            return pd.DataFrame()
        df = df.dropna(subset=["Close"])
        return df
    except Exception as e:
        st.error(f"خطا در دریافت داده‌ها: {e}")
        return pd.DataFrame()

data = load_data(symbol)

# بررسی وجود داده
if data.empty:
    st.warning("⚠️ داده‌ای برای نمایش وجود ندارد. ممکن است API یا اینترنت مشکل داشته باشد.")
else:
    # نمایش جدول
    st.subheader(f"📈 قیمت {coin_name} در ۳۰ روز گذشته")
    st.dataframe(data.tail())

    # رسم نمودار
    fig = px.line(
        data.reset_index(),
        x="Date",
        y="Close",
        title=f"روند قیمت {coin_name}",
        labels={"Close": "قیمت پایانی", "Date": "تاریخ"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # تغییرات ۲۴ ساعته
    last = data["Close"].iloc[-1]
    prev = data["Close"].iloc[-2] if len(data) > 1 else last
    change24 = ((last - prev) / prev * 100) if prev != 0 else 0.0
    st.metric(label="تغییرات ۲۴ ساعته", value=f"{change24:.2f} %", delta=f"{change24:.2f} %")
