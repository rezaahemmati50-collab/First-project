import yfinance as yf
import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Crypto & Stock Prediction Dashboard", layout="wide")
st.title("📊 Crypto & Stock Multi-Prediction Dashboard")

# --- 1. ورودی‌ها
tickers = st.text_input("Enter tickers (comma separated)", "BTC-USD, ETH-USD, AAPL")
period = st.selectbox("Select historical period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
days_ahead = st.slider("Days to forecast", min_value=7, max_value=90, value=30)
refresh_data = st.checkbox("🔄 Auto-refresh every 1 minute", value=False)

if refresh_data:
    st.experimental_rerun()

# --- 2. رسم نمودار ترکیبی قیمت‌ها
if tickers:
    tickers_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    combined_fig = go.Figure()

    for ticker in tickers_list:
        try:
            df = yf.download(ticker, period=period, interval="1d")

            if 'Close' not in df.columns or df.empty:
                st.warning(f"⚠️ No valid 'Close' price found for {ticker}. Skipping...")
                continue

            df = df.reset_index()
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Close'])

            combined_fig.add_trace(go.Scatter(
                x=df['Date'], y=df['Close'], mode='lines', name=ticker
            ))

        except Exception as e:
            st.error(f"❌ Error fetching data for {ticker}: {e}")

    st.subheader("📉 Combined Price Chart")
    combined_fig.update_layout(title="Historical Prices", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(combined_fig, use_container_width=True)

# --- 3. پیش‌بینی جداگانه برای هر ارز/سهام
    for ticker in tickers_list:
        try:
            df = yf.download(ticker, period=period, interval="1d")

            if 'Close' not in df.columns or df.empty:
                st.warning(f"⚠️ No prediction: No 'Close' price for {ticker}.")
                continue

            df = df.reset_index()
            df = df.rename(columns={"Date": "ds", "Close": "y"})
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df = df.dropna()

            if df.empty:
                st.warning(f"⚠️ Skipping {ticker} due to missing data.")
                continue

            m = Prophet(daily_seasonality=True)
            m.fit(df)

            future = m.make_future_dataframe(periods=days_ahead)
            forecast = m.predict(future)

            st.subheader(f"🔮 Prediction for {ticker}")
            fig = plot_plotly(m, forecast)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error in prediction for {ticker}: {e}")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
