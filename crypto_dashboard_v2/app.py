import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

# تنظیمات صفحه
st.set_page_config(page_title="داشبورد پیشرفته ارز دیجیتال", layout="wide")
st.title("📈 داشبورد پیشرفته ارز دیجیتال")

# انتخاب ارز
crypto_symbol = st.selectbox(
    "یک ارز دیجیتال انتخاب کنید:",
    ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "DOGE-USD"]
)
crypto_name = crypto_symbol.split("-")[0]

# دریافت داده‌ها
@st.cache_data
def load_data(symbol):
    df = yf.download(symbol, period="30d", interval="1h")
    df.dropna(inplace=True)
    return df

data = load_data(crypto_symbol)

# بخش قیمت و تغییرات
st.subheader("📊 قیمت و تغییرات")
if not data.empty and len(data) >= 2:
    last = data["Close"].iloc[-1]
    prev = data["Close"].iloc[-2]
    change24 = (last - prev) / prev * 100 if prev != 0 else 0.0
    st.metric(label=f"قیمت پایانی {crypto_name}", value=f"${last:,.2f}", delta=f"{change24:.2f}%")
else:
    st.warning("داده کافی برای نمایش قیمت وجود ندارد.")

# نمودار قیمت
st.subheader("📉 نمودار قیمت 30 روز گذشته")
if not data.empty:
    fig = px.line(data, x=data.index, y="Close", title=f"{crypto_name} - 30 روز گذشته")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("داده‌ای برای نمایش نمودار موجود نیست.")

# پیش‌بینی قیمت با Prophet
st.subheader("🔮 پیش‌بینی 3 روز آینده")
if not data.empty:
    df_forecast = data.reset_index()[["Datetime", "Close"]]
    df_forecast.rename(columns={"Datetime": "ds", "Close": "y"}, inplace=True)

    model = Prophet(daily_seasonality=True)
    model.fit(df_forecast)

    future = model.make_future_dataframe(periods=72, freq='H')
    forecast = model.predict(future)

    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)
else:
    st.warning("داده کافی برای پیش‌بینی وجود ندارد.")

# جدول داده‌ها
st.subheader("📋 داده‌های اخیر")
st.dataframe(data.tail(20))
