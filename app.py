import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet

# عنوان برنامه
st.set_page_config(page_title="تحلیل و پیش‌بینی ارز دیجیتال", layout="wide")
st.title("📈 داشبورد تحلیل و پیش‌بینی ارز دیجیتال")

# انتخاب ارز و بازه زمانی
col1, col2 = st.columns(2)
with col1:
    ticker = st.selectbox("انتخاب ارز دیجیتال:", ["BTC-USD", "ETH-USD", "ADA-USD", "XLM-USD"])
with col2:
    period = st.selectbox("بازه زمانی:", ["1d", "5d", "1wk", "2wk"], index=1)

# دریافت داده‌ها
try:
    data = yf.download(ticker, period=period, interval="1h")
    if data.empty:
        st.error("داده‌ای برای این بازه زمانی یافت نشد.")
        st.stop()
except Exception as e:
    st.error(f"خطا در دریافت داده: {e}")
    st.stop()

# محاسبه میانگین‌های متحرک
data["MA20"] = data["Close"].rolling(window=20).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()

# نمایش آخرین قیمت
latest_price = float(data["Close"].dropna().iloc[-1])
st.metric(label="💰 آخرین قیمت", value=f"${latest_price:,.2f}")

# تعیین سیگنال خرید/فروش
if len(data.dropna()) >= 50:
    ma20 = float(data["MA20"].dropna().iloc[-1])
    ma50 = float(data["MA50"].dropna().iloc[-1])
    if latest_price > ma20 > ma50:
        signal = "📈 خرید"
    elif latest_price < ma20 < ma50:
        signal = "📉 فروش"
    else:
        signal = "🤝 نگهداری"
    st.subheader(f"سیگنال معاملاتی: {signal}")
else:
    st.warning("داده کافی برای محاسبه سیگنال وجود ندارد.")

# رسم نمودار قیمت و MA
st.line_chart(data[["Close", "MA20", "MA50"]])

# پیش‌بینی با Prophet
df = data.reset_index()[["Date", "Close"]]
df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

pred_3d, pred_7d = None, None

if df["y"].dropna().shape[0] >= 2:
    try:
        m = Prophet()
        m.fit(df)

        future = m.make_future_dataframe(periods=7)
        forecast = m.predict(future)

        if len(forecast) >= 4:
            pred_3d = forecast["yhat"].iloc[-7 + 3]
        if len(forecast) >= 7:
            pred_7d = forecast["yhat"].iloc[-1]

        # نمایش پیش‌بینی‌ها
        col3, col4 = st.columns(2)
        with col3:
            if pred_3d:
                st.metric("پیش‌بینی ۳ روز بعد", f"${pred_3d:,.2f}")
        with col4:
            if pred_7d:
                st.metric("پیش‌بینی ۷ روز بعد", f"${pred_7d:,.2f}")

        # رسم نمودار پیش‌بینی
        st.subheader("📊 نمودار پیش‌بینی قیمت")
        fig_forecast = m.plot(forecast)
        st.pyplot(fig_forecast)

    except Exception as e:
        st.error(f"خطا در پیش‌بینی: {e}")
else:
    st.warning("داده کافی برای پیش‌بینی وجود ندارد.")
