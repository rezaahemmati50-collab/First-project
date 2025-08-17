import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Dashboard", page_icon="🟡", layout="wide")

# ----------------- CoinGecko API -----------------
def fetch_from_coingecko(symbol: str, vs_currency="usd", days=90):
    try:
        # symbol مثل BTC-USD → ما فقط BTC رو میخوایم
        sym = symbol.replace("-USD", "").lower()
        url = f"https://api.coingecko.com/api/v3/coins/{sym}/market_chart"
        params = {"vs_currency": vs_currency, "days": days, "interval": "daily"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            prices = data.get("prices", [])
            if not prices:
                return pd.DataFrame()
            df = pd.DataFrame(prices, columns=["timestamp", "price"])
            df["Date"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.rename(columns={"price": "Close"}, inplace=True)
            return df[["Date", "Close"]]
    except Exception as e:
        st.warning(f"CoinGecko error for {symbol}: {e}")
    return pd.DataFrame()

# ----------------- Yahoo Finance -----------------
def fetch_from_yahoo(symbol: str, period="3mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df.rename(columns={"Adj Close": "Close"}, inplace=True)
        return df[["Date", "Close"]]
    except Exception:
        return pd.DataFrame()

# ----------------- Unified fetch -----------------
@st.cache_data(ttl=300)
def fetch_crypto(symbol: str, period="3mo", interval="1d"):
    df = fetch_from_yahoo(symbol, period, interval)
    if df.empty:
        st.info(f"No data from Yahoo Finance for {symbol}, trying CoinGecko...")
        df = fetch_from_coingecko(symbol, days=90)
    return df

# ----------------- Main -----------------
st.title("🟡 Global Crypto Dashboard")
st.caption("Live prices with Yahoo Finance + CoinGecko fallback")

symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "ENA-USD"]

rows = []
for s in symbols:
    df = fetch_crypto(s)
    if df.empty:
        rows.append({"Symbol": s, "Price": None, "Change24h": None})
        continue

    last = float(df["Close"].iloc[-1])
    prev = float(df["Close"].iloc[-2]) if df.shape[0] > 1 else last
    change = (last - prev) / prev * 100 if prev != 0 else 0.0

    rows.append({"Symbol": s, "Price": last, "Change24h": round(change, 2)})

df_out = pd.DataFrame(rows)
if not df_out.empty:
    st.dataframe(df_out, hide_index=True, use_container_width=True)
else:
    st.error("No data available from Yahoo or CoinGecko.")
