import streamlit as st
import requests
import pandas as pd

st.title("Crypto from CoinGecko")

url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
params = {"vs_currency": "usd", "days": "7"}  # 7 روز گذشته
r = requests.get(url, params=params)

if r.status_code == 200:
    data = r.json()
    prices = data["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    st.line_chart(df.set_index("timestamp")["price"])
else:
    st.error("CoinGecko API not responding")
