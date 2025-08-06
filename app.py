import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from prophet import Prophet

st.set_page_config(page_title="تحلیل بازار ارز دیجیتال", layout="centered")

# دریافت داده از یاهو فایننس
def get_data(symbol):
    return yf.download(symbol, period="3mo", interval="1d")

# تحلیل تکنیکال با RSI و MACD
def generate_signal(data):
    if data.empty or 'Close' not in data.columns:
        return "⚠️ داده‌ای برای تحلیل وجود ندارد"

    close = data['Close'].ffill()
    close = pd.Series(close.values.flatten(), index=close.index)

    try:
        rsi = ta.momentum.RSIIndicator(close).rsi()
        macd = ta.trend.MACD(close).macd_diff()

        last_rsi = rsi.iloc[-1]
        last_macd = macd.iloc[-1]

        if last_rsi < 30 and last_macd > 0:
            return "🔵 سیگنال خرید (Buy)"
        elif last_rsi > 70 and last_macd < 0:
            return "🔴 سیگنال فروش (Sell)"
        else:
            return "🟡 نگه‌داری (Hold)"
    except Exception as e:
        return f"⚠️ خطا در محاسبه اندیکاتورها: {e}"

# پیش‌بینی قیمت با Prophet
def predict_with_prophet(data, days):
    df = data[['Close']].copy().reset_index()
    df.columns = ['ds', 'y']

    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(subset=['ds', 'y'], inplace=True)

    if not isinstance(df['y'], pd.Series) or df['y'].ndim != 1:
        raise ValueError("ستون 'y' باید یک Series یک‌بعدی باشد.")

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    predicted = forecast[['ds', 'yhat']].tail(days)
    return predicted

# رابط کاربری
st.title("📊 تحلیل و پیش‌بینی بازار ارز دیجیتال")

assets = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "Stellar (XLM)": "XLM-USD",
    "Solana (SOL)": "SOL-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Ripple (XRP)": "XRP-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Polkadot (DOT)": "DOT-USD"
}

asset_name = st.selectbox("🪙 انتخاب ارز دیجیتال:", list(assets.keys()) + ["🔤 وارد کردن دستی..."])

if asset_name == "🔤 وارد کردن دستی...":
    custom_symbol = st.text_input("نماد ارز دلخواه را وارد کنید (مثلاً SHIB-USD):")
    symbol = custom_symbol.strip().upper()
else:
    symbol = assets[asset_name]

# دریافت داده‌ها
if symbol:
    with st.spinner("⏳ در حال دریافت داده‌ها..."):
        data = get_data(symbol)

    if data.empty:
        st.error("⚠️ داده‌ای برای این نماد پیدا نشد. لطفاً نماد را بررسی کنید.")
        st.stop()

    st.subheader("📈 نمودار قیمت")
    st.line_chart(data['Close'])

    st.subheader("📌 سیگنال تحلیل تکنیکال")
    signal = generate_signal(data)
    st.markdown(f"### {signal}")

    # پیش‌بینی قیمت
    st.subheader("🤖 پیش‌بینی قیمت با Prophet")
    forecast_days = st.selectbox("⏱ بازه زمانی پیش‌بینی:", [3, 7, 30], format_func=lambda x: f"{x} روز آینده")

    try:
        predicted_df = predict_with_prophet(data, days=forecast_days)
        predicted_df['yhat'] = predicted_df['yhat'].round(2)
        predicted_df['ds'] = predicted_df['ds'].dt.date
        predicted_df.columns = ['تاریخ', 'قیمت پیش‌بینی‌شده (USD)']
        st.table(predicted_df)
    except Exception as e:
        st.error(f"⚠️ خطا در پیش‌بینی قیمت: {e}")
