import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# لیست ارزها
coins = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "ADA-USD": "Cardano",
    "XLM-USD": "Stellar",
    "BNB-USD": "Binance Coin",
    "XRP-USD": "Ripple",
    "SOL-USD": "Solana",
    "DOGE-USD": "Dogecoin",
    "DOT-USD": "Polkadot"
}

st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("📊 داشبورد تحلیل ارزهای دیجیتال (روش دوم - Moving Average)")

results = []

for ticker, name in coins.items():
    try:
        data = yf.download(ticker, period="2d", interval="1h")

        if data.empty:
            results.append({
                "ارز": name,
                "قیمت فعلی": "—",
                "تغییر ۲۴ساعته (%)": "—",
                "پیشنهاد": "—"
            })
            continue

        # قیمت فعلی
        latest_price = data["Close"].iloc[-1]

        # تغییرات ۲۴ ساعته
        if len(data) >= 25:
            price_24h_ago = data["Close"].iloc[-25]
            change_24h = ((latest_price - price_24h_ago) / price_24h_ago) * 100
        else:
            change_24h = None

        # میانگین متحرک کوتاه‌مدت و بلندمدت
        short_ma = data["Close"].rolling(window=5).mean().iloc[-1]
        long_ma = data["Close"].rolling(window=20).mean().iloc[-1]

        if np.isnan(short_ma) or np.isnan(long_ma):
            suggestion = "داده کافی نیست"
        elif short_ma > long_ma:
            suggestion = "📈 خرید"
        elif short_ma < long_ma:
            suggestion = "📉 فروش"
        else:
            suggestion = "⏸ نگه‌داری"

        results.append({
            "ارز": name,
            "قیمت فعلی": round(latest_price, 2),
            "تغییر ۲۴ساعته (%)": round(change_24h, 2) if change_24h is not None else "—",
            "پیشنهاد": suggestion
        })

    except Exception as e:
        results.append({
            "ارز": name,
            "قیمت فعلی": "—",
            "تغییر ۲۴ساعته (%)": "—",
            "پیشنهاد": "خطا"
        })

# نمایش جدول
df = pd.DataFrame(results)
st.dataframe(df, use_container_width=True)

# نمایش نمودار قیمت
st.subheader("📉 نمودار قیمت")
coin_selected = st.selectbox("یک ارز انتخاب کنید", list(coins.values()))
ticker_selected = [k for k, v in coins.items() if v == coin_selected][0]

chart_data = yf.download(ticker_selected, period="7d", interval="1h")
if not chart_data.empty:
    chart_data = chart_data[["Close"]]
    chart_data.columns = ["قیمت"]
    st.line_chart(chart_data)
else:
    st.warning("داده‌ای برای نمایش نمودار یافت نشد.")
