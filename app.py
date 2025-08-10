import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import date, timedelta

# -------------------------------
# تنظیمات صفحه
# -------------------------------
st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("📊 داشبورد پیش‌بینی و تحلیل ارز دیجیتال")

# -------------------------------
# انتخاب ارز و بازه زمانی
# -------------------------------
crypto_list = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Cardano": "ADA-USD",
    "XRP": "XRP-USD",
    "Stellar": "XLM-USD"
}

col1, col2 = st.columns(2)
with col1:
    crypto_name = st.selectbox("ارز دیجیتال را انتخاب کنید:", list(crypto_list.keys()))
with col2:
    days_ahead = st.slider("تعداد روزهای پیش‌بینی:", 1, 10, 5)

symbol = crypto_list[crypto_name]

# -------------------------------
# دریافت داده‌ها
# -------------------------------
end_date = date.today()
start_date = end_date - timedelta(days=365)

data = yf.download(symbol, start=start_date, end=end_date)

if data.empty:
    st.error("❌ داده‌ای برای این ارز یافت نشد. لطفاً ارز دیگری انتخاب کنید.")
    st.stop()

# اطمینان از اینکه ستون Close موجود است
if 'Close' not in data.columns:
    st.error("ستون قیمت بسته شدن (Close) در داده‌ها یافت نشد.")
    st.stop()

# -------------------------------
# نمایش جدول
# -------------------------------
st.subheader("📅 داده‌های اخیر بازار")
st.dataframe(data.tail(10))

# -------------------------------
# ترسیم نمودار قیمت
# -------------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='قیمت بسته شدن',
    line=dict(color='cyan')
))
fig.update_layout(
    title=f"نمودار قیمت {crypto_name}",
    xaxis_title="تاریخ",
    yaxis_title="قیمت (USD)",
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# پیش‌بینی ساده (با میانگین متحرک)
# -------------------------------
data['MA_7'] = data['Close'].rolling(window=7).mean()
last_price = data['Close'].iloc[-1]
future_dates = [end_date + timedelta(days=i) for i in range(1, days_ahead+1)]
predicted_prices = [last_price * (1 + (i * 0.01)) for i in range(days_ahead)]

pred_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Price": predicted_prices
})

st.subheader("📈 پیش‌بینی قیمت")
st.dataframe(pred_df)

pred_fig = go.Figure()
pred_fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='تاریخچه قیمت',
    line=dict(color='lightblue')
))
pred_fig.add_trace(go.Scatter(
    x=pred_df['Date'],
    y=pred_df['Predicted Price'],
    mode='lines+markers',
    name='پیش‌بینی',
    line=dict(color='orange', dash='dash')
))
pred_fig.update_layout(
    title="پیش‌بینی قیمت در روزهای آینده",
    xaxis_title="تاریخ",
    yaxis_title="قیمت (USD)",
    template="plotly_dark"
)
st.plotly_chart(pred_fig, use_container_width=True)
