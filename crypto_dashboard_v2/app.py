# app.py
# Global Crypto Insight — Stable Version (No cache)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

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
        first_col = df.columns[0]
        df.rename(columns={first_col: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    expected = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    return df[["Date"] + expected].dropna(subset=["Date"]).sort_values("Date")

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
    s = pd.to_numeric(close_s, errors="coerce").dropna()
    if s.empty:
        return np.array([np.nan]*days)
    last = float(s.iloc[-1])
    avg_pct = s.pct_change().dropna().mean() if s.shape[0] > 1 else 0.0
    return np.array([last * ((1 + (avg_pct if np.isfinite(avg_pct) else 0.0)) ** i) for i in range(1, days+1)])

# ---------- UI ----------
st.title("Global Crypto Insight (Stable)")

TOP = ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD","ENA-USD"]

currency = st.sidebar.selectbox("Currency", ["USD","CAD","EUR"], index=0)
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)

primary = st.sidebar.selectbox("Primary symbol", TOP, index=0)
extra = st.sidebar.text_input("Extra symbols (comma separated)", value="DOGE-USD,MATIC-USD")
extra_list = [s.strip().upper() for s in extra.split(",") if s.strip()]

symbols = []
for s in [primary] + TOP + extra_list:
    if s not in symbols:
        symbols.append(s)

# ---------- Market Overview ----------
st.header("Market Overview")

summary = []
for s in symbols[:10]:
    d = fetch_yf(s, period=period, interval=interval)
    c = series_close(d).dropna()
    if c.empty:
        summary.append({"Symbol": s, "Price": None, "Change24h": None})
        continue
    price_usd = float(c.iloc[-1])
    prev_usd = float(c.iloc[-2]) if c.shape[0] >= 2 else price_usd
    change24 = (price_usd - prev_usd) / prev_usd * 100 if prev_usd != 0 else 0.0
    summary.append({"Symbol": s, "Price": price_usd, "Change24h": round(change24, 2)})

df_sum = pd.DataFrame(summary)
if not df_sum.empty:
    df_sum["PriceStr"] = df_sum["Price"].apply(lambda v: f"{v:,.2f} {currency}" if pd.notna(v) else "—")
    df_sum["ChangeStr"] = df_sum["Change24h"].apply(lambda v: f"{v:+.2f}%")
    st.dataframe(df_sum[["Symbol", "PriceStr", "ChangeStr"]], hide_index=True)
else:
    st.info("No data.")

# ---------- Forecast ----------
st.header("Forecast")
f_sym = st.selectbox("Choose symbol", options=symbols, index=0)
df_f = fetch_yf(f_sym, period=period, interval=interval)
cs = series_close(df_f).dropna()
if not cs.empty:
    st.line_chart(df_f.set_index("Date")[cs.name])
    arr = moving_avg_forecast(cs, 7)
    last_date = df_f["Date"].iloc[-1]
    fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(7)]
    fc_table = pd.DataFrame({"Date": fc_dates, "Forecast": np.round(arr, 2)})
    st.table(fc_table)
else:
    st.warning("No data for this symbol.")
