import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from datetime import datetime, timedelta

# عنوان برنامه
st.title("📊 داشبورد تحلیل و پیش‌بینی قیمت ارز دیجیتال")

# انتخاب ارز دیجیتال
crypto_symbol = st.selectbox(
    "ارز دیجیتال را انتخاب کنید:",
    ["BTC-USD", "ETH-USD", "ADA-USD", "XLM-USD"]
)

# انتخاب بازه زمانی
period = st.selectbox(
    "بازه زمانی:",
    {
        "1 روز": "1d",
        "5 روز": "5d",
        "1 هفته": "1wk",
        "2 هفته": "2wk"
    }
)

# دانلود داده‌ها
data = yf.download(crypto_symbol, period=period)

# میانگین‌های متحرک
if not data.empty:
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()

    # آخرین قیمت
    if not data["Close"].dropna().empty:
        latest_price = data["Close"].dropna().iloc[-1]
        st.metric(label="💰 آخرین قیمت", value=f"${latest_price:,.2f}")
    else:
        st.warning("قیمت پایانی برای این بازه موجود نیست.")

    # نمایش نمودار
    st.line_chart(data[["Close", "MA20", "MA50"]].dropna())

    # پیش‌بینی قیمت با Prophet
    forecast_days = st.selectbox("مدت پیش‌بینی:", [3, 7])  # ۳ روز یا ۷ روز
    df = data.reset_index()[["Date", "Close"]]
    df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

    if len(df.dropna()) >= 2:  # حداقل دو داده برای مدل
        m = Prophet(daily_seasonality=True)
        m.fit(df)

        future = m.make_future_dataframe(periods=forecast_days)
        forecast = m.predict(future)

        st.subheader("📈 پیش‌بینی قیمت")
        forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days)
        st.dataframe(forecast_display)

        # نمودار پیش‌بینی
        forecast_chart = pd.DataFrame({
            "Date": forecast["ds"],
            "پیش‌بینی": forecast["yhat"],
            "حد بالا": forecast["yhat_upper"],
            "حد پایین": forecast["yhat_lower"]
        })
        st.line_chart(forecast_chart.set_index("Date"))
    else:
        st.warning("داده کافی برای پیش‌بینی وجود ندارد.")
else:
    st.error("هیچ داده‌ای برای این بازه یافت نشد.")
