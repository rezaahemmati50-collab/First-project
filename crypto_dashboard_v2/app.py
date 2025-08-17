import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------- Page setup ----------
st.set_page_config(page_title="Crypto Dashboard", page_icon="🟡", layout="wide")

pd.options.mode.chained_assignment = None

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
        first_col = df.columns[0]
        df.rename(columns={first_col: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    expected = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    return df[["Date"] + expected].dropna(subset=["Date"]).sort_values("Date")

@st.cache_data(ttl=180)
def fetch_yf(symbol: str, period="3mo", interval="1d") -> pd.DataFrame:
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
    col = "Close" if "Close" in df.columns else None
    if col is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

def moving_avg_forecast(close_s: pd.Series, days: int) -> np.ndarray:
    try:
        s = pd.to_numeric(close_s, errors="coerce").dropna()
        if s.empty:
            return np.array([np.nan]*days)
        last = float(s.iloc[-1])
        avg_pct = s.pct_change().dropna().mean() if s.shape[0] > 1 else 0.0
        return np.array([last * ((1 + (avg_pct if np.isfinite(avg_pct) else 0.0)) ** i) for i in range(1, days+1)])
    except Exception:
        return np.array([np.nan]*days)

def compute_combined_signal(close_series, next_forecast=None):
    cs = pd.to_numeric(close_series, errors="coerce").dropna()
    if cs.empty:
        return ("NO DATA", "#9e9e9e")
    ma20 = cs.rolling(20, min_periods=1).mean().iloc[-1]
    ma50 = cs.rolling(50, min_periods=1).mean().iloc[-1]
    score = 0
    if ma20 > ma50: score += 1
    if next_forecast is not None and np.isfinite(next_forecast):
        if next_forecast > cs.iloc[-1]: score += 1
    if score >= 2: return ("STRONG BUY", "#b2ff66")
    if score == 1: return ("BUY", "#ffe36b")
    return ("SELL", "#ff7b7b")

# ---------- Sidebar ----------
st.sidebar.header("Settings")

SYMBOLS = ["BTC-USD","ETH-USD","ADA-USD","SOL-USD","ENA-USD"]

currency = st.sidebar.selectbox("Display currency", ["USD","CAD","EUR","GBP"], index=0)
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)

# ---------- Market Tab ----------
st.header("Market Overview")

summary = []
for s in SYMBOLS:
    df = fetch_yf(s, period=period, interval=interval)
    if df.empty:
        st.warning(f"No data from Yahoo Finance for {s}")
        continue
    c = series_close(df).dropna()
    if c.empty:
        st.warning(f"No valid Close data for {s}")
        continue
    price = float(c.iloc[-1])
    prev = float(c.iloc[-2]) if c.shape[0] >= 2 else price
    change24 = (price - prev) / prev * 100 if prev != 0 else 0.0
    fc1 = moving_avg_forecast(c, 1)
    next_fc = float(fc1[0]) if len(fc1) > 0 else None
    label, color = compute_combined_signal(c, next_fc)
    summary.append({
        "Symbol": s,
        "Price": round(price, 2),
        "Change24h": round(change24, 2),
        "Signal": label
    })

if summary:
    df_sum = pd.DataFrame(summary)
    st.dataframe(df_sum, use_container_width=True, hide_index=True)
else:
    st.info("No data available for selected symbols.")
