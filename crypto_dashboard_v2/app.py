# app.py — Global Crypto Insight (Safe Version)
# Fix for missing 'Close' errors & empty data handling

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Global Crypto Insight", page_icon="🟡", layout="wide")

# -------- Optional libraries --------
try:
    from prophet import Prophet
    HAS_PROPHET = True
except:
    HAS_PROPHET = False

try:
    import ta
    HAS_TA = True
except:
    HAS_TA = False

# -------- Helpers --------
def ensure_flat_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

@st.cache_data(ttl=180)
def fetch_yf(symbol: str, period="3mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = ensure_flat_columns(df).reset_index()
        if 'Date' not in df.columns:
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except:
        return pd.DataFrame()

def moving_avg_forecast(series, days):
    s = series.dropna()
    if s.empty:
        return np.array([np.nan]*days)
    last = s.iloc[-1]
    avg_pct = s.pct_change().dropna().mean() if len(s) > 1 else 0
    return np.array([last * ((1 + avg_pct) ** i) for i in range(1, days+1)])

def fmt_currency(x, cur="USD"):
    try:
        return f"{x:,.2f} {cur}"
    except:
        return "—"

# -------- Sidebar --------
st.sidebar.header("Settings")
currency = st.sidebar.selectbox("Display currency", ["USD","CAD","EUR","GBP"], 0)
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y","2y"], 1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], 0)

symbols_input = st.sidebar.text_area("Symbols", value="BTC-USD,ETH-USD,ADA-USD").upper()
symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

# FX rate
@st.cache_data(ttl=300)
def get_fx(target):
    if target == "USD":
        return 1.0
    mp = {"CAD": "USDCAD=X", "EUR": "USDEUR=X", "GBP": "USDGBP=X"}
    t = mp.get(target)
    if not t:
        return 1.0
    df = yf.download(t, period="5d", interval="1d", progress=False)
    if df.empty:
        return 1.0
    return float(df['Close'].dropna().iloc[-1])

fx = get_fx(currency)

tabs = st.tabs(["Market", "Forecast"])
tab_market, tab_forecast = tabs

# -------- Market Tab --------
with tab_market:
    st.header("Market Overview")
    summary = []
    for s in symbols:
        df = fetch_yf(s, period=period, interval=interval)
        if df.empty or 'Close' not in df.columns:
            summary.append({"Symbol": s, "Price": None, "Change24h": None})
            continue
        df = df.dropna(subset=["Close"])
        if df.empty:
            summary.append({"Symbol": s, "Price": None, "Change24h": None})
            continue
        last = float(df['Close'].iloc[-1]) * fx
        prev = float(df['Close'].iloc[-2]) * fx if len(df) > 1 else last
        ch = (last - prev) / prev * 100 if prev != 0 else 0
        summary.append({"Symbol": s, "Price": last, "Change24h": ch})
    st.dataframe(pd.DataFrame(summary))

# -------- Forecast Tab --------
with tab_forecast:
    st.header("Forecast")
    f_sym = st.selectbox("Choose symbol", options=symbols)
    if f_sym:
        df_f = fetch_yf(f_sym, period=period, interval=interval)
        if df_f.empty or 'Close' not in df_f.columns:
            st.warning("No data for forecast.")
        else:
            df_f = df_f.dropna(subset=["Close"])
            if df_f.empty:
                st.warning("No valid Close prices.")
            else:
                arr = moving_avg_forecast(df_f['Close'], 7)
                last_date = df_f['Date'].iloc[-1]
                fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(len(arr))]
                st.table(pd.DataFrame({"Date": fc_dates, "Forecast": arr}))
