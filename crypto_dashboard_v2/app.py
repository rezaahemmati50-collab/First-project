# app.py
# Global Crypto Insight — Safe version (no infinite loading)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------- Page setup ----------
st.set_page_config(page_title="Global Crypto Insight", page_icon="🟡", layout="wide")

pd.options.mode.chained_assignment = None

# ---------- Helpers ----------
def fake_data(symbol="BTC-USD", days=60):
    """Generate fake OHLC data if Yahoo Finance fails"""
    dates = pd.date_range(end=datetime.today(), periods=days, freq="D")
    base = 20000 if "BTC" in symbol else 1000
    prices = base + np.cumsum(np.random.randn(days) * (base * 0.01))
    df = pd.DataFrame({
        "Date": dates,
        "Open": prices,
        "High": prices * (1 + np.random.rand(days) * 0.02),
        "Low": prices * (1 - np.random.rand(days) * 0.02),
        "Close": prices,
        "Volume": np.random.randint(1000, 10000, size=days)
    })
    return df

def normalize_ohlc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date")

@st.cache_data(ttl=180)
def fetch_yf(symbol: str, period="3mo", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            st.warning(f"No data from Yahoo Finance for {symbol}, using fake data.")
            return fake_data(symbol)
        return normalize_ohlc_index(df)
    except Exception:
        st.warning(f"Yahoo Finance error for {symbol}, using fake data.")
        return fake_data(symbol)

def series_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Close" in df.columns:
        return pd.to_numeric(df["Close"], errors="coerce")
    return pd.Series(dtype=float)

# ---------- UI ----------
st.title("🟡 Global Crypto Insight (Safe Mode)")
st.caption("Loads even if Yahoo Finance is down (uses fake data fallback).")

symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "ENA-USD"]

tab1, tab2 = st.tabs(["Market", "Forecast"])

# ---------- Market Tab ----------
with tab1:
    st.header("Market Overview")

    summary = []
    for s in symbols:
        df = fetch_yf(s, period="3mo", interval="1d")
        cs = series_close(df).dropna()
        if cs.empty:
            summary.append({"Symbol": s, "Price": None, "Change24h": None})
            continue
        last = cs.iloc[-1]
        prev = cs.iloc[-2] if len(cs) >= 2 else last
        change = (last - prev) / prev * 100 if prev != 0 else 0
        summary.append({"Symbol": s, "Price": last, "Change24h": round(change, 2)})

    df_sum = pd.DataFrame(summary)
    if not df_sum.empty:
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
    else:
        st.info("No summary data.")

# ---------- Forecast Tab ----------
with tab2:
    st.header("Simple Forecast")
    f_sym = st.selectbox("Choose symbol", options=symbols, index=0)
    horizon = st.slider("Forecast days", 3, 30, 7)

    df = fetch_yf(f_sym, period="6mo", interval="1d")
    cs = series_close(df).dropna()

    if cs.empty:
        st.warning("No data available for forecast.")
    else:
        last = float(cs.iloc[-1])
        avg_pct = cs.pct_change().mean()
        forecast = [last * ((1 + avg_pct) ** i) for i in range(1, horizon + 1)]
        dates = [df["Date"].iloc[-1] + timedelta(days=i) for i in range(1, horizon + 1)]

        st.line_chart(pd.DataFrame({"History": cs, "Forecast": pd.Series(forecast, index=dates)}))
