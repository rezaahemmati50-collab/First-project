import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from prophet import Prophet
from prophet.plot import plot_plotly

st.set_page_config(page_title="Crypto Dashboard", layout="wide")
st.title("📊 داشبورد تحلیل و پیش‌بینی ارزهای دیجیتال")

# لیست ارزها
cryptos = {
    "بیت‌کوین (BTC)": "BTC-USD",
    "اتریوم (ETH)": "ETH-USD",
    "کاردانو (ADA)": "ADA-USD",
    "ریپل (XRP)": "XRP-USD",
    "استلار (XLM)": "XLM-USD"
}
crypto_name = st.selectbox("یک ارز انتخاب کنید:", list(cryptos.keys()))
symbol = cryptos[crypto_name]

period = st.selectbox("بازه زمانی:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)

# دریافت داده‌ها
data = yf.download(symbol, period=period)
data = data.dropna()

if data.empty or "Close" not in data.columns:
    st.error("❌ داده معتبر برای این ارز پیدا نشد.")
else:
    # محاسبه تغییرات
    if len(data) >= 2:
        last = float(data["Close"].iloc[-1])
        prev = float(data["Close"].iloc[-2])
        change24 = (last - prev) / prev * 100 if prev != 0 else 0.0
    else:
        last = prev = change24 = 0.0

    st.metric(label=f"📈 قیمت پایانی {crypto_name}", value=f"${last:,.2f}", delta=f"{change24:.2f}%")

    # نمودار قیمت
    if not data["Close"].isna().all():
        fig = px.line(
            data.reset_index(),
            x="Date",
            y="Close",
            title=f"روند قیمت {crypto_name}",
            labels={"Close": "قیمت پایانی", "Date": "تاریخ"}
        )
        st.plotly_chart(fig, use_container_width=True)

    # پیش‌بینی قیمت
    st.subheader("🔮 پیش‌بینی قیمت آینده")
    df_prophet = data.reset_index()[["Date", "Close"]]
    df_prophet.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=30)  # پیش‌بینی ۳۰ روز آینده
    forecast = model.predict(future)

    fig_forecast = plot_plotly(model, forecast)
    st.plotly_chart(fig_forecast, use_container_width=True)

    # نمایش داده‌ها
    st.subheader("📋 داده‌های خام")
    st.dataframe(data.tail(10))
