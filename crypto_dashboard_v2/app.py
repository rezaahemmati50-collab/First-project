import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Crypto Test", page_icon="💹", layout="wide")
st.title("Crypto Test App")

# انتخاب کوین (فقط تستی)
symbol = "BTC-USD"
st.write(f"Fetching data for: {symbol}")

try:
    df = yf.download(symbol, period="1mo", interval="1d", progress=False)
    if df.empty:
        st.error(f"No data from Yahoo Finance for {symbol}")
    else:
        st.success("Data fetched successfully ✅")
        st.write("Last rows:")
        st.dataframe(df.tail())
        st.line_chart(df["Close"])
except Exception as e:
    st.error(f"Error fetching data: {e}")
