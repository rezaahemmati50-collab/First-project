import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet

st.set_page_config(page_title="تحلیل ارز دیجیتال", layout="wide")

st.title("📊 داشبورد تحلیل ارز دیجیتال")

# انتخاب ارز و بازه
symbol = st.selectbox("ارز مورد نظر را انتخاب کنید:", ["BTC-USD", "ETH-USD", "ADA-USD", "XLM-USD"])
period = st.selectbox("بازه زمانی:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])

# دریافت داده
data = yf.download(symbol, period=period)

if data.empty or data["Close"].dropna().empty:
    st.warning("⚠️ هیچ داده‌ای برای این بازه یافت نشد. لطفاً بازه یا ارز دیگری انتخاب کنید.")
else:
    # محاسبه میانگین متحرک
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()

    # آخرین قیمت
    latest_price = data["Close"].dropna().iloc[-1]
    if pd.isna(latest_price):
        st.warning("⚠️ قیمت معتبر یافت نشد.")
    else:
        st.metric(label="💰 آخرین قیمت", value=f"${float(latest_price):,.2f}")

    # نمایش نمودار
    available_cols = [col for col in ["Close", "MA20", "MA50"] if col in data.columns]
    if available_cols:
        st.line_chart(data[available_cols])
    else:
        st.warning("⚠️ ستون‌های کافی برای نمایش نمودار یافت نشد.")

    # پیش‌بینی با Prophet
    df = data.reset_index()[["Date", "Close"]].dropna()
    df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

    if len(df) >= 2:  # Prophet حداقل 2 سطر نیاز دارد
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)
        st.subheader("🔮 پیش‌بینی قیمت ۳۰ روز آینده")
        st.line_chart(forecast[["ds", "yhat"]].set_index("ds"))
    else:
        st.warning("⚠️ داده کافی برای پیش‌بینی وجود ندارد.")
