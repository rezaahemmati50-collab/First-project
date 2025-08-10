import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import date

# 📌 تنظیمات اولیه صفحه
st.set_page_config(page_title="Crypto Forecast", layout="wide")
st.title("📈 پیش‌بینی قیمت ارز دیجیتال")

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# 📌 انتخاب ارز دیجیتال
coins = ("BTC-USD", "ADA-USD", "XLM-USD", "ETH-USD")
selected_coin = st.selectbox("یک ارز انتخاب کنید:", coins)

# 📌 انتخاب تایم‌فریم
interval = st.selectbox("تایم‌فریم داده‌ها:", ("1d", "1wk", "1mo"))

# 📌 انتخاب تعداد روز پیش‌بینی
n_days = st.slider("تعداد روزهای پیش‌بینی:", 1, 365)
period = n_days

# 📌 بارگذاری داده‌ها با کش
@st.cache_data
def load_data(ticker, interval):
    data = yf.download(ticker, START, TODAY, interval=interval)
    data.reset_index(inplace=True)
    return data

# 🚀 بارگذاری دیتا
data_load_state = st.text("در حال بارگذاری داده‌ها...")
df = load_data(selected_coin, interval)
data_load_state.text("✅ داده‌ها بارگذاری شدند!")

# 📌 نمایش داده‌های خام
st.subheader("📊 داده‌های خام")
st.dataframe(df.tail())

# 📌 نمودار قیمت و حجم معاملات
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="قیمت بسته شدن"))
fig_price.update_layout(title="نمودار قیمت", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_price)

fig_vol = go.Figure()
fig_vol.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="حجم معاملات"))
fig_vol.update_layout(title="نمودار حجم معاملات")
st.plotly_chart(fig_vol)

# 📌 آماده‌سازی دیتا برای Prophet
df_train = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')

# 📌 مدل Prophet
m = Prophet()
m.fit(df_train)

# 📌 ایجاد پیش‌بینی
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# 📌 نمایش پیش‌بینی
st.subheader("📈 پیش‌بینی قیمت آینده")
fig_forecast = plot_plotly(m, forecast)
st.plotly_chart(fig_forecast)

# 📌 نمایش جزئیات پیش‌بینی
st.subheader("🔍 داده‌های پیش‌بینی")
st.dataframe(forecast.tail())

# 📌 دکمه دانلود CSV پیش‌بینی
csv = forecast.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 دانلود پیش‌بینی به صورت CSV",
    data=csv,
    file_name=f"{selected_coin}_forecast.csv",
    mime="text/csv",
)

st.success("برنامه با موفقیت اجرا شد 🚀")
