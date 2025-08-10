import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date

# 🎯 تنظیمات اولیه
st.set_page_config(page_title="Crypto Forecast App", layout="wide")

st.title("📈 پیش‌بینی قیمت ارز دیجیتال با Prophet")

START = "2016-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# انتخاب ارز
cryptos = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "Stellar (XLM)": "XLM-USD",
    "Solana (SOL)": "SOL-USD"
}

selected_crypto = st.selectbox("یک ارز انتخاب کنید:", list(cryptos.keys()))
n_years = st.slider("تعداد سال‌های پیش‌بینی:", 1, 4)
period = n_years * 365

# 📥 دریافت داده‌ها
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("در حال دریافت داده‌ها...")
df = load_data(cryptos[selected_crypto])
data_load_state.text("✅ داده‌ها با موفقیت بارگذاری شدند!")

# نمایش داده‌ها
st.subheader("📊 داده‌های خام")
st.write(df.tail())

# 📌 آماده‌سازی دیتا برای Prophet با بررسی ایمنی
if 'Date' not in df.columns:
    df = df.reset_index()

if 'Close' not in df.columns:
    st.error("❌ ستون Close در داده‌ها یافت نشد. لطفاً تایم‌فریم یا ارز دیگری انتخاب کنید.")
    st.stop()

df_train = pd.DataFrame({
    "ds": pd.to_datetime(df['Date'], errors='coerce'),
    "y": pd.to_numeric(df['Close'], errors='coerce')
})

df_train = df_train.dropna()

if df_train.empty:
    st.error("❌ داده معتبر برای آموزش مدل وجود ندارد.")
    st.stop()

# 📈 مدل Prophet
m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# نمایش پیش‌بینی
st.subheader("📈 پیش‌بینی قیمت")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# اجزای پیش‌بینی
st.subheader("📉 اجزای پیش‌بینی")
fig2 = m.plot_components(forecast)
st.write(fig2)
