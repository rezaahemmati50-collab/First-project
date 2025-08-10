import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

# عنوان برنامه
st.set_page_config(page_title="Crypto Signal Dashboard", layout="wide")
st.title("📊 Crypto Buy/Sell Signal Dashboard")

# انتخاب ارز
cryptos = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "Ripple (XRP)": "XRP-USD",
    "Stellar (XLM)": "XLM-USD"
}
crypto_name = st.selectbox("ارز دیجیتال را انتخاب کنید:", list(cryptos.keys()))
symbol = cryptos[crypto_name]

# انتخاب بازه زمانی
days = st.slider("تعداد روزهای گذشته برای نمایش:", 30, 365, 180)

# دانلود داده
end_date = date.today()
start_date = end_date - timedelta(days=days)
data = yf.download(symbol, start=start_date, end=end_date)

# بررسی ستون قیمت
price_col = None
for col in ["Close", "Adj Close"]:
    if col in data.columns:
        price_col = col
        break

if price_col is None or data.empty:
    st.error("❌ داده‌ای برای این ارز پیدا نشد. لطفا ارز یا بازه زمانی دیگری انتخاب کنید.")
    st.stop()

# محاسبه میانگین متحرک
data["MA20"] = data[price_col].rolling(window=20).mean()
data["MA50"] = data[price_col].rolling(window=50).mean()

# سیگنال خرید / فروش
latest_price = data[price_col].iloc[-1]
ma20 = data["MA20"].iloc[-1]
ma50 = data["MA50"].iloc[-1]

if latest_price > ma20 > ma50:
    signal = "🟢 خرید"
elif latest_price < ma20 < ma50:
    signal = "🔴 فروش"
else:
    signal = "🟡 نگه‌داری"

# نمایش سیگنال
st.markdown(f"### سیگنال برای {crypto_name}: **{signal}**")
st.markdown(f"**قیمت فعلی:** ${latest_price:,.2f}")

# نمودار
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data[price_col], mode='lines', name='قیمت'))
fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], mode='lines', name='MA20'))
fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], mode='lines', name='MA50'))

fig.update_layout(
    title=f"{crypto_name} Price Chart",
    xaxis_title="تاریخ",
    yaxis_title="قیمت (USD)",
    template="plotly_dark",
    legend=dict(x=0, y=1)
)

st.plotly_chart(fig, use_container_width=True)

# جدول داده‌ها
st.subheader("📅 داده‌های قیمتی")
st.dataframe(data.tail(20))
