import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet

st.set_page_config(page_title="تحلیل ارز دیجیتال", layout="wide")

# لیست ارزها
coins = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "Stellar (XLM)": "XLM-USD"
}

# انتخاب ارز
coin_name = st.selectbox("انتخاب ارز دیجیتال:", list(coins.keys()))
coin_symbol = coins[coin_name]

# انتخاب بازه زمانی
period = st.selectbox("بازه زمانی:", ["1d", "3d", "7d", "14d", "1mo", "3mo", "6mo", "1y"])

# دریافت داده
data = yf.download(coin_symbol, period=period, interval="1d")
if data.empty:
    st.error("داده‌ای برای این بازه زمانی موجود نیست.")
    st.stop()

# محاسبه میانگین‌های متحرک
data["MA20"] = data["Close"].rolling(window=20).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()

# آخرین قیمت و میانگین‌ها
latest_price = data["Close"].dropna().iloc[-1] if not data["Close"].dropna().empty else None
ma20 = data["MA20"].dropna().iloc[-1] if not data["MA20"].dropna().empty else None
ma50 = data["MA50"].dropna().iloc[-1] if not data["MA50"].dropna().empty else None

# پیش‌بینی با Prophet
df = data.reset_index()[["Date", "Close"]]
df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=7)
forecast = m.predict(future)

# گرفتن پیش‌بینی 3 روز و 7 روز آینده
pred_3d = forecast["yhat"].iloc[-7 + 3] if len(forecast) >= 3 else None
pred_7d = forecast["yhat"].iloc[-1] if len(forecast) >= 7 else None

# نمایش قیمت‌ها
col1, col2, col3 = st.columns(3)
col1.metric("آخرین قیمت", f"${latest_price:,.2f}" if pd.notna(latest_price) else "نامشخص")
col2.metric("پیش‌بینی 3 روز آینده", f"${pred_3d:,.2f}" if pd.notna(pred_3d) else "نامشخص")
col3.metric("پیش‌بینی 7 روز آینده", f"${pred_7d:,.2f}" if pd.notna(pred_7d) else "نامشخص")

# سیگنال خرید/فروش
if pd.notna(latest_price) and pd.notna(ma20) and pd.notna(ma50):
    if latest_price > ma20 > ma50:
        st.success("📈 سیگنال: خرید")
    elif latest_price < ma20 < ma50:
        st.error("📉 سیگنال: فروش")
    else:
        st.info("⏳ سیگنال: نگه‌داری")
else:
    st.warning("داده کافی برای محاسبه سیگنال وجود ندارد.")

# نمایش نمودار قیمت
st.line_chart(data["Close"])
