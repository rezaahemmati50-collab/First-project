# app.py — Global Crypto Insight (Stable Version)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# ---------- Page setup ----------
st.set_page_config(page_title="Global Crypto Insight", page_icon="🟡", layout="wide")

# ---------- Helpers ----------
def ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def normalize_ohlc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = ensure_flat_columns(df).reset_index()
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    expected = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    return df[["Date"] + expected].dropna(subset=["Date"]).sort_values("Date")

@st.cache_data(ttl=300)
def fetch_yf(symbol: str, period="6mo", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return normalize_ohlc_index(df)
    except Exception:
        return pd.DataFrame()

def series_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Close" not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df["Close"], errors="coerce")

def moving_avg_forecast(close_s: pd.Series, days: int) -> np.ndarray:
    s = pd.to_numeric(close_s, errors="coerce").dropna()
    if s.empty:
        return np.array([np.nan]*days)
    last = float(s.iloc[-1])
    avg_pct = s.pct_change().dropna().mean() if s.shape[0] > 1 else 0.0
    return np.array([last * ((1 + avg_pct) ** i) for i in range(1, days+1)])

# ---------- Sidebar ----------
st.sidebar.header("Settings")

SYMBOLS = [
    "BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD",
    "DOGE-USD","ADA-USD","ENA-USD"  # ENA added, might have limited data
]

currency = st.sidebar.selectbox("Display currency", ["USD","CAD","EUR","GBP"], index=0)
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)
primary = st.sidebar.selectbox("Primary symbol", SYMBOLS, index=0)

# ---------- Tabs ----------
tabs = st.tabs(["Market","Forecast","About"])
tab_market, tab_forecast, tab_about = tabs

# ---------- Market ----------
with tab_market:
    st.header("Market Overview")
    summary = []
    for s in SYMBOLS:
        df = fetch_yf(s, period=period, interval=interval)
        cs = series_close(df).dropna()
        if cs.empty:
            summary.append({"Symbol": s, "Price": None, "Change24h": None, "Signal": "NO DATA"})
            continue
        last = cs.iloc[-1]
        prev = cs.iloc[-2] if len(cs) > 1 else last
        change24 = (last - prev) / prev * 100 if prev != 0 else 0.0
        summary.append({"Symbol": s, "Price": last, "Change24h": round(change24,2), "Signal": "CHECK"})

    df_sum = pd.DataFrame(summary)
    if not df_sum.empty:
        df_sum["PriceStr"] = df_sum["Price"].apply(lambda v: f"{v:,.2f} {currency}" if pd.notna(v) else "—")
        df_sum["ChangeStr"] = df_sum["Change24h"].apply(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
        st.dataframe(df_sum[["Symbol","PriceStr","ChangeStr","Signal"]], hide_index=True, use_container_width=True)
    else:
        st.warning("⚠️ No Yahoo data returned for any symbol. Try another period/interval.")

# ---------- Forecast ----------
with tab_forecast:
    st.header("Forecast (Moving Avg)")
    f_sym = st.selectbox("Choose symbol", options=SYMBOLS, index=0)
    df_f = fetch_yf(f_sym, period=period, interval=interval)
    cs = series_close(df_f).dropna()
    if df_f.empty or cs.empty:
        st.warning(f"⚠️ No Yahoo data for {f_sym} in {period}/{interval}. Try another setting.")
    else:
        horizon = st.slider("Forecast horizon (days)", 3, 30, 7)
        arr = moving_avg_forecast(cs, horizon)
        last_date = df_f["Date"].iloc[-1]
        fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(horizon)]
        st.line_chart(pd.DataFrame({"History": cs, "Forecast": pd.Series(arr, index=fc_dates)}))

# ---------- About ----------
with tab_about:
    st.write("**Global Crypto Insight (stable)**")
    st.write("- Uses Yahoo Finance")
    st.write("- Includes BTC, ETH, BNB, SOL, XRP, DOGE, ADA, ENA")
    st.write("- Moving Average forecast model")
    st.caption("⚠️ Educational only, not financial advice.")
