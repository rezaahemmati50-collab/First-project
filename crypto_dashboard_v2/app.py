import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Crypto Test", page_icon="💹", layout="wide")
st.title("Crypto Quick Test")

symbol = "BTC-USD"
st.write(f"Fetching data for: {symbol}")

try:
    df = yf.download(symbol, period="5d", interval="1d", progress=False, timeout=5)
    if df is None or df.empty:
        st.warning(f"⚠️ No data received for {symbol}")
    else:
        st.success("✅ Data fetched successfully")
        st.dataframe(df.tail())
        st.line_chart(df["Close"])
except Exception as e:
    st.error(f"Error: {e}")
