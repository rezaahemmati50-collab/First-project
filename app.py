import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.set_page_config(page_title="Crypto Signal App", page_icon="📈", layout="centered")

# -----------------------------
# توابع کمکی
# -----------------------------

def get_data(symbol):
    data = yf.download(symbol, period="3mo", interval="1d")
    return data

def generate_signal(data):
    # محاسبه اندیکاتورها
    rsi = ta.momentum.RSIIndicator(data['Close']).rsi()
    macd = ta.trend.MACD(data['Close']).macd_diff()

    # آخرین مقدارها
    last_rsi = rsi.iloc[-1]
    last_macd = macd.iloc[-1]

    # منطق ساده سیگنال
    if last_rsi < 30 and last_macd > 0:
        return "🔵 خرید (Buy)"
    elif last_rsi > 70 and last_macd < 0:
        return "🔴 فروش (Sell)"
    else:
        return "🟡 نگه‌داری (Hold)"

# -----------------------------
# رابط کاربری
# -----------------------------

st.title("📈 پیشنهاد خرید/فروش ارز دیجیتال")

assets = {
    "Bitcoin (BTC)": "BTC-USD",
    "Cardano (ADA)": "ADA-USD",
    "Stellar (XLM)": "XLM-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD"
}

asset_name = st.selectbox("✅ انتخاب ارز:", list(assets.keys()))
symbol = assets[asset_name]

with st.spinner("در حال دریافت اطلاعات..."):
    data = get_data(symbol)

# نمایش نمودار
st.line_chart(data['Close'])

# نمایش سیگنال
signal = generate_signal(data)
st.subheader(f"📊 سیگنال پیشنهادی برای {asset_name}:")
st.markdown(f"### {signal}")
