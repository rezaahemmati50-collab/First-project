import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from datetime import date

st.set_page_config(page_title="تحلیل ارز دیجیتال", layout="wide")

st.title("📈 داشبورد تحلیل و پیش‌بینی ارز دیجیتال")

# انتخاب ارز
crypto_symbol = st.selectbox(
    "ارز مورد نظر را انتخاب کنید",
    ["BTC-USD", "ETH-USD", "ADA-USD", "XLM-USD"]
)

# انتخاب بازه زمانی
period_map = {
    "1 روز": "1d",
    "5 روز": "5d",
    "1 هفته": "1wk",
    "2 هفته": "2wk",
}
period_choice = st.selectbox("بازه زمانی", list(period_map.keys()))
period = period_map[period_choice]

# دانلود داده
data = yf.download(crypto_symbol, period=period, interval="1h" if period in ["1d", "5d"] else "1d")

if data.empty:
    st.error("داده‌ای برای این بازه زمانی موجود نیست.")
else:
    # محاسبه میانگین متحرک
    if len(data) >= 20:
        data["MA20"] = data["Close"].rolling(window=20).mean()
    if len(data) >= 50:
        data["MA50"] = data["Close"].rolling(window=50).mean()

    # آخرین قیمت
    latest_price = data["Close"].iloc[-1]
    st.metric(label="💰 آخرین قیمت", value=f"${latest_price:,.2f}")

    # 📊 نمایش نمودار با ستون‌های موجود
    available_cols = [col for col in ["Close", "MA20", "MA50"] if col in data.columns]
    if available_cols:
        st.line_chart(data[available_cols])
    else:
        st.warning("داده کافی برای نمایش نمودار وجود ندارد.")

    # 📅 پیش‌بینی قیمت با Prophet
    st.subheader("🔮 پیش‌بینی قیمت")
    forecast_days = st.selectbox("مدت پیش‌بینی", [3, 7])
    df = data.reset_index()[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

    if len(df.dropna()) >= 2:
        m = Prophet(daily_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)

        st.line_chart(forecast.set_index("ds")[["yhat", "yhat_lower", "yhat_upper"]])
    else:
        st.warning("داده کافی برای پیش‌بینی موجود نیست.")
