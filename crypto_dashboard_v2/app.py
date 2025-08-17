import streamlit as st
import requests
import pandas as pd

st.title("Crypto Price Test")

try:
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "7"}
    headers = {"User-Agent": "Mozilla/5.0"}  # جلوگیری از بلاک شدن
    r = requests.get(url, params=params, headers=headers, timeout=10)

    if r.status_code == 200:
        data = r.json()
        prices = data.get("prices", [])
        if prices:
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            st.line_chart(df.set_index("timestamp")["price"])
        else:
            st.warning("No price data received.")
    else:
        st.error(f"API error: {r.status_code}")

except Exception as e:
    st.error(f"Request failed: {e}")
