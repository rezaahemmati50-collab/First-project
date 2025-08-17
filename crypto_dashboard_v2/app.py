import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Dashboard", page_icon="🟡", layout="wide")

# ---------- Fake fetch ----------
def fake_data(symbol="BTC-USD", days=90):
    dates = pd.date_range(end=datetime.today(), periods=days)
    price = np.cumsum(np.random.randn(days)) + 30000  # random walk
    df = pd.DataFrame({"Date": dates, "Close": price})
    return df

# ---------- Sidebar ----------
st.sidebar.header("Settings")
symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "ENA-USD"]
symbol = st.sidebar.selectbox("Select coin", symbols, index=0)

# ---------- Main ----------
st.header("Market Overview")

df = fake_data(symbol)
st.line_chart(df.set_index("Date")["Close"])

st.write("Latest price:", round(df["Close"].iloc[-1], 2))
