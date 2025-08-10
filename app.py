import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date

# عنوان برنامه
st.title("📈 پیش‌بینی قیمت ارز دیجیتال")

# انتخاب ارز
cryptos = ["BTC-USD", "ETH-USD", "ADA-USD", "XLM-USD"]
selected_crypto = st.selectbox("انتخاب ارز دیجیتال:", cryptos)

# انتخاب تعداد روزهای پیش‌بینی
n_days = st.slider("تعداد روزهای پیش‌بینی", 1, 365, 30)

# محدوده تاریخی
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# بارگذاری داده
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

st.subheader("📥 در حال دریافت داده‌ها...")
df = load_data(selected_crypto)

# بررسی ستون‌ها
if 'Close' not in df.columns:
    st.error("❌ ستون Close در داده‌ها یافت نشد. لطفاً ارز یا تایم‌فریم دیگری انتخاب کنید.")
    st.stop()

if df['Close'].empty:
    st.error("❌ ستون Close خالی است. داده‌ای برای پردازش وجود ندارد.")
    st.stop()

# آماده‌سازی داده برای Prophet
y_values = pd.to_numeric(df['Close'], errors='coerce')

if y_values.isna().all():
    st.error("❌ همه مقادیر ستون Close نامعتبر هستند.")
    st.stop()

df_train = pd.DataFrame({
    "ds": pd.to_datetime(df['Date'], errors='coerce'),
    "y": y_values
}).dropna()

# رسم نمودار اولیه
st.subheader("📊 نمودار قیمت")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="قیمت بسته شدن"))
fig.layout.update(xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# مدل Prophet
model = Prophet()
model.fit(df_train)

future = model.make_future_dataframe(periods=n_days)
forecast = model.predict(future)

# نمایش پیش‌بینی
st.subheader("🔮 پیش‌بینی قیمت")
fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast)

st.write("📅 داده‌های پیش‌بینی شده:")
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
