import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

# عنوان برنامه
st.set_page_config(page_title="داشبورد ارز دیجیتال", layout="wide")
st.title("📊 داشبورد تحلیل قیمت ارزهای دیجیتال")

# لیست ارزها
cryptos = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Cardano": "ADA-USD",
    "Stellar": "XLM-USD"
}

# انتخاب ارز
crypto_name = st.selectbox("یک ارز انتخاب کنید:", list(cryptos.keys()))
crypto_symbol = cryptos[crypto_name]

# بازه زمانی
days = st.slider("تعداد روزهای گذشته:", min_value=7, max_value=365, value=90)

# دریافت داده
end_date = datetime.today()
start_date = end_date - timedelta(days=days)

data = yf.download(crypto_symbol, start=start_date, end=end_date)

# نمایش داده به صورت جدول
if not data.empty:
    st.subheader(f"داده‌های {crypto_name}")
    st.dataframe(data.tail())
else:
    st.warning("⚠️ داده‌ای برای این ارز پیدا نشد.")

# رسم نمودار فقط اگر داده معتبر بود
if not data.empty and "Close" in data.columns:
    fig = px.line(data, x=data.index, y="Close", title=f"قیمت پایانی {crypto_name}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error(f"❌ داده یا ستون 'Close' برای {crypto_name} پیدا نشد.")

# محاسبه تغییرات
if not data.empty and "Close" in data.columns:
    change = data["Close"].pct_change().mean() * 100
    st.metric(label="میانگین درصد تغییرات روزانه", value=f"{change:.2f}%")
