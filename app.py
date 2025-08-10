import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from datetime import date

# -----------------------------
# تنظیمات اولیه
# -----------------------------
START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("📈 پیش‌بینی قیمت ارز دیجیتال با Prophet")

# -----------------------------
# انتخاب ارز دیجیتال
# -----------------------------
cryptos = ("BTC-USD", "ETH-USD", "ADA-USD", "XLM-USD")
selected_crypto = st.selectbox("ارز دیجیتال را انتخاب کنید:", cryptos)

# -----------------------------
# انتخاب مدت پیش‌بینی
# -----------------------------
n_years = st.slider("مدت پیش‌بینی (سال)", 1, 4)
period = n_years * 365

# -----------------------------
# دریافت داده‌ها
# -----------------------------
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("در حال دریافت داده‌ها...")
data = load_data(selected_crypto)
data_load_state.text("✅ داده‌ها با موفقیت بارگذاری شدند!")

st.subheader("نمایش داده‌ها")
st.write(data.tail())

# -----------------------------
# پیش‌پردازش برای Prophet
# -----------------------------
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# چک کردن ستون y
if 'y' in df_train.columns:
    if isinstance(df_train['y'], (pd.Series, list)):
        df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
        df_train = df_train.dropna(subset=['y'])
    else:
        st.error("❌ ستون y فرمت درستی ندارد.")
        st.stop()
else:
    st.error("❌ ستون y پیدا نشد.")
    st.stop()

if df_train.empty:
    st.error("❌ داده‌ای برای آموزش Prophet پیدا نشد.")
    st.stop()

# -----------------------------
# مدل Prophet
# -----------------------------
m = Prophet()
m.fit(df_train)

# -----------------------------
# پیش‌بینی آینده
# -----------------------------
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# -----------------------------
# نمایش نتایج
# -----------------------------
st.subheader("پیش‌بینی قیمت")
st.write(forecast.tail())

fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader("ترکیب اجزای پیش‌بینی")
fig2 = m.plot_components(forecast)
st.write(fig2)
