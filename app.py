import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import matplotlib.pyplot as plt

st.set_page_config(page_title="Crypto Forecast", layout="wide")
st.title("💹 داشبورد پیش‌بینی ارز دیجیتال")

# انتخاب ارز
cryptos = {
    "Bitcoin": "BTC-USD",
    "Cardano": "ADA-USD",
    "Stellar": "XLM-USD",
    "Ethereum": "ETH-USD",
    "Litecoin": "LTC-USD"
}
crypto_name = st.selectbox("ارز را انتخاب کنید:", list(cryptos.keys()))
symbol = cryptos[crypto_name]

# انتخاب بازه زمانی
period = st.selectbox("بازه زمانی داده‌ها:", ["1y", "2y", "5y", "max"], index=0)

# دانلود داده
st.info("📥 در حال دریافت داده‌ها...")
df = yf.download(symbol, period=period)
df.reset_index(inplace=True)

if df.empty:
    st.error("❌ داده‌ای برای این ارز پیدا نشد.")
    st.stop()

# نمایش داده
st.subheader("📊 داده‌های قیمتی")
st.dataframe(df.tail())

# Prophet Model
st.subheader("🔮 پیش‌بینی با Prophet")
df_train = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

if 'y' in df_train.columns and not df_train.empty:
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
    df_train = df_train.dropna(subset=['y'])

    if df_train.empty:
        st.error("❌ داده‌ای برای آموزش Prophet پیدا نشد. لطفاً بازه یا ارز را تغییر دهید.")
    else:
        m = Prophet()
        m.fit(df_train)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)

        fig1 = m.plot(forecast)
        st.pyplot(fig1)

        st.subheader("📅 پیش‌بینی ۳۰ روز آینده")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))
else:
    st.error("❌ داده یا ستون y وجود ندارد.")
