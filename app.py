import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet

st.set_page_config(page_title="تحلیل ارز دیجیتال", layout="wide")
st.title("📊 داشبورد تحلیل ارز دیجیتال")

# انتخاب ارز و بازه زمانی
symbol = st.selectbox("ارز مورد نظر را انتخاب کنید:", ["BTC-USD", "ETH-USD", "ADA-USD", "XLM-USD"])
period = st.selectbox("بازه زمانی:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

# دریافت داده
data = yf.download(symbol, period=period)

# اگر ستون‌ها چندسطحی (MultiIndex) بودند، ساده‌شان می‌کنیم
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

# بررسی داده
if data.empty or "Close" not in data.columns or data["Close"].dropna().empty:
    st.warning("⚠️ داده‌ای یافت نشد.")
else:
    # محاسبه میانگین متحرک
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()

    # آخرین قیمت
    latest_price = float(data["Close"].dropna().iloc[-1])
    st.metric(label="💰 آخرین قیمت", value=f"${latest_price:,.2f}")

    # نمایش نمودار قیمت و میانگین‌ها
    st.line_chart(data[["Close", "MA20", "MA50"]])

    # پیش‌بینی با Prophet
    df = data.reset_index()[["Date", "Close"]].dropna()
    df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

    if len(df) >= 2:
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        st.subheader("🔮 پیش‌بینی قیمت ۳۰ روز آینده")
        st.line_chart(forecast[["ds", "yhat"]].set_index("ds"))
    else:
        st.warning("⚠️ داده کافی برای پیش‌بینی وجود ندارد.")
