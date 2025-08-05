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
    # بررسی اینکه دیتا خالی نباشه
    if data.empty:
        return "⚠️ داده‌ای برای تحلیل وجود ندارد"

    # بررسی وجود ستون Close
    if 'Close' not in data.columns:
        return "⚠️ ستون Close در داده‌ها موجود نیست"

    close = data['Close'].ffill()

    # بررسی اگر تمام مقادیر Close خالی باشند
    if close.isnull().all():
        return "⚠️ مقادیر Close خالی هستند"

    try:
        # محاسبه اندیکاتورها
        rsi = ta.momentum.RSIIndicator(close).rsi()
        macd = ta.trend.MACD(close).macd_diff()

        # آخرین مقادیر
        last_rsi = rsi.iloc[-1]
        last_macd = macd.iloc[-1]

        # منطق سیگنال‌دهی ساده
        if last_rsi < 30 and last_macd > 0:
            return "🔵 خرید (Buy)"
        elif last_rsi > 70 and last_macd < 0:
            return "🔴 فروش (Sell)"
        else:
            return "🟡 نگه‌داری (Hold)"
    except Exception as e:
        return f"⚠️ خطا در محاسبه اندیکاتورها: {e}"

# -----------------------------
# رابط کاربری Streamlit
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

# بررسی خالی بودن دیتا
if data.empty:
    st.error("⚠️ داده‌ای برای این ارز پیدا نشد.")
    st.stop()

# نمایش نمودار قیمت
st.line_chart(data['Close'])

# دریافت سیگنال
signal = generate_signal(data)
st.subheader(f"📊 سیگنال پیشنهادی برای {asset_name}:")
st.markdown(f"### {signal}")
