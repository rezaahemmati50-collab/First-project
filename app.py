import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date

# --- عنوان ---
st.set_page_config(page_title="Crypto Price Forecast", layout="wide")
st.title("📊 Crypto Price Analysis & Forecast")

# --- انتخاب ارز ---
crypto_list = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'XLM-USD']
selected_crypto = st.selectbox("Select a cryptocurrency:", crypto_list)

# --- تعداد روزهای پیش‌بینی ---
n_days = st.slider("Days of prediction:", min_value=1, max_value=30, value=7)

# --- دانلود داده‌ها ---
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, period="2y")
    data.reset_index(inplace=True)
    return data

data = load_data(selected_crypto)

st.subheader("Raw Data")
st.dataframe(data.tail())

# --- رسم قیمت‌ها ---
st.subheader("Price Chart")
st.line_chart(data[['Date', 'Close']].set_index('Date'))

# --- آماده‌سازی داده برای Prophet ---
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# --- مدل ---
m = Prophet(daily_seasonality=True)
m.fit(df_train)

future = m.make_future_dataframe(periods=n_days)
forecast = m.predict(future)

# --- نمایش پیش‌بینی ---
st.subheader(f"Forecast for next {n_days} days")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# --- جدول پیش‌بینی ---
st.subheader("Forecast Data")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days))

# --- آخرین قیمت و پیش‌بینی ---
latest_price = round(data['Close'].iloc[-1], 2)
predicted_price = round(forecast['yhat'].iloc[-1], 2)

st.metric(label="Latest Price (USD)", value=f"${latest_price}")
st.metric(label=f"Predicted Price in {n_days} days", value=f"${predicted_price}")
