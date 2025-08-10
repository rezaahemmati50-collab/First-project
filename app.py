import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import date

# عنوان
st.set_page_config(page_title="Crypto Price Prediction", layout="wide")
st.title("💹 پیش‌بینی و تحلیل بازار ارز دیجیتال")

# انتخاب ارز
cryptos = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Cardano (ADA-USD)": "ADA-USD",
    "Ripple (XRP-USD)": "XRP-USD"
}
coin = st.selectbox("ارز دیجیتال مورد نظر را انتخاب کنید:", list(cryptos.keys()))
symbol = cryptos[coin]

# محدوده تاریخ
start_date = st.date_input("تاریخ شروع", value=pd.to_datetime("2023-01-01"))
end_date = date.today()

# دریافت داده
data = yf.download(symbol, start=start_date, end=end_date)
data.reset_index(inplace=True)

st.subheader("📊 داده‌های تاریخی")
st.dataframe(data.tail())

# نمودار قیمت بسته شدن
fig_close = go.Figure()
fig_close.add_trace(go.Scatter(
    x=data['Date'],
    y=data['Close'].tolist(),
    name="قیمت بسته شدن",
    line=dict(color='royalblue', width=2)
))
fig_close.update_layout(
    title="نمودار قیمت بسته شدن",
    xaxis_title="تاریخ",
    yaxis_title="قیمت (USD)",
    template="plotly_dark"
)
st.plotly_chart(fig_close, use_container_width=True)

# مدل Prophet برای پیش‌بینی
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
model = Prophet(daily_seasonality=True)
model.fit(df_train)

# پیش‌بینی چند روز آینده
n_days = st.slider("تعداد روزهای پیش‌بینی", 1, 30, 7)
future = model.make_future_dataframe(periods=n_days)
forecast = model.predict(future)

# نمایش پیش‌بینی
st.subheader("📈 پیش‌بینی قیمت")
fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

# نمایش داده‌های پیش‌بینی
st.subheader("📄 جدول پیش‌بینی")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(n_days))
