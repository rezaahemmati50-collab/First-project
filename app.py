import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime

st.set_page_config(page_title="Crypto Signal App", layout="wide")

# 🎯 لیست ارزها
cryptos = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "XRP": "XRP-USD",
    "Stellar (XLM)": "XLM-USD"
}

# 🖱 انتخاب ارز و بازه زمانی
crypto_name = st.sidebar.selectbox("ارز مورد نظر را انتخاب کنید", list(cryptos.keys()))
symbol = cryptos[crypto_name]
period = st.sidebar.selectbox("بازه زمانی", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

# 📥 دریافت داده‌ها
data = yf.download(symbol, period=period, interval="1d")
data["MA20"] = data["Close"].rolling(window=20).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()

# 🔍 بررسی آخرین قیمت و سیگنال
latest_price = float(data["Close"].iloc[-1])
ma20_ready = not pd.isna(data["MA20"].iloc[-1])
ma50_ready = not pd.isna(data["MA50"].iloc[-1])

if ma50_ready:
    ma20 = float(data["MA20"].iloc[-1])
    ma50 = float(data["MA50"].iloc[-1])
    if latest_price > ma20 > ma50:
        signal, color = "📈 خرید", "green"
    elif latest_price < ma20 < ma50:
        signal, color = "📉 فروش", "red"
    else:
        signal, color = "⏳ نگه‌داری", "orange"
elif ma20_ready:
    ma20 = float(data["MA20"].iloc[-1])
    if latest_price > ma20:
        signal, color = "📈 خرید (بر اساس MA20)", "green"
    elif latest_price < ma20:
        signal, color = "📉 فروش (بر اساس MA20)", "red"
    else:
        signal, color = "⏳ نگه‌داری (بر اساس MA20)", "orange"
else:
    signal, color = "⚠️ داده کافی برای محاسبه میانگین‌ها وجود ندارد", "gray"

# 📢 نمایش سیگنال
st.markdown(f"<h2 style='color:{color}'>سیگنال برای {crypto_name}: {signal}</h2>", unsafe_allow_html=True)

# 📊 نمودار کندل استیک + MA
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name='Candlestick'
))
fig.add_trace(go.Scatter(x=data.index, y=data["MA20"], mode='lines', name='MA20', line=dict(color='blue', width=1.5)))
fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], mode='lines', name='MA50', line=dict(color='purple', width=1.5)))

fig.update_layout(title=f"📊 نمودار {crypto_name}",
                  xaxis_title="تاریخ",
                  yaxis_title="قیمت (USD)",
                  template="plotly_dark",
                  xaxis_rangeslider_visible=False)

st.plotly_chart(fig, use_container_width=True)

# 📥 دکمه دانلود داده
csv = data.to_csv().encode('utf-8')
st.download_button(
    label="📥 دانلود داده‌ها (CSV)",
    data=csv,
    file_name=f"{symbol}_{datetime.now().strftime('%Y-%m-%d')}.csv",
    mime='text/csv'
)
