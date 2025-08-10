import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# ----------------- عنوان
st.title("📊 داشبورد تحلیل و پیش‌بینی ارز دیجیتال")

# ----------------- ورودی‌ها
ticker = st.text_input("نماد ارز دیجیتال (مثلاً BTC-USD)", "BTC-USD")
years = st.slider("چند سال آینده را پیش‌بینی کنیم؟", 1, 4)
period = years * 365

# ----------------- دانلود داده‌ها
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2018-01-01")
    df.reset_index(inplace=True)
    return df

data_load_state = st.text("در حال دانلود داده‌ها...")
df = load_data(ticker)
data_load_state.text("✅ داده‌ها با موفقیت بارگذاری شدند.")

# ----------------- بررسی ستون Close
if 'Close' not in df.columns or df['Close'].isnull().all():
    st.error("❌ ستون Close پیدا نشد یا خالی است. لطفاً نماد را بررسی کنید.")
    st.stop()

# اطمینان از اینکه Close عددی است
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df = df.dropna(subset=['Close'])

# ----------------- نمایش داده‌ها
st.subheader("📈 آخرین داده‌ها")
st.write(df.tail())

# ----------------- رسم نمودار قیمت
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close Price"))
fig.layout.update(title_text="نمودار قیمت", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# ----------------- پیش‌بینی با Prophet
train_df = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(train_df)

future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader("🔮 پیش‌بینی قیمت")
fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast)

# ----------------- جدول پیش‌بینی
st.subheader("📄 داده‌های پیش‌بینی")
st.write(forecast.tail())
