# app.py
# Global Crypto Insight — Fixed Version (safe Change24h + no cache)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
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
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
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
    try:
        s = pd.to_numeric(close_s, errors="coerce").dropna()
        if s.empty:
            return np.array([np.nan]*days)
        last = float(s.iloc[-1])
        avg_pct = s.pct_change().dropna().mean() if s.shape[0] > 1 else 0.0
        return np.array([last * ((1 + (avg_pct if np.isfinite(avg_pct) else 0.0)) ** i) for i in range(1, days+1)])
    except Exception:
        return np.array([np.nan]*days)

def fmt_currency(x, cur="USD"):
    try:
        return f"{x:,.2f} {cur}"
    except Exception:
        return "—"

# ---------- Sidebar ----------
st.sidebar.header("Settings")

TOP_COINS = [
    "BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD",
    "ADA-USD","DOGE-USD","ENA-USD"  # اضافه شده ENA
]

currency = st.sidebar.selectbox("Display currency", ["USD","CAD","EUR","GBP"], index=0)
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)

primary = st.sidebar.selectbox("Primary symbol", TOP_COINS, index=0)

symbols = [primary] + [s for s in TOP_COINS if s != primary]

# ---------- Tabs ----------
tabs = st.tabs(["Market","Forecast"])
tab_market, tab_forecast = tabs

# ---------- Market Tab ----------
with tab_market:
    st.header("Market Overview")

    summary = []
    for s in symbols:
        d = fetch_yf(s, period=period, interval=interval)
        c = series_close(d).dropna()
        if c.empty:
            summary.append({"Symbol": s, "Price": None, "Change24h": None})
            continue
        price = float(c.iloc[-1])
        prev = float(c.iloc[-2]) if c.shape[0] >= 2 else price
        change24 = (price - prev) / prev * 100 if prev != 0 else 0.0
        summary.append({
            "Symbol": s,
            "Price": price,
            "Change24h": change24
        })

    df_sum = pd.DataFrame(summary)
    if not df_sum.empty:
        df_sum["PriceStr"] = df_sum["Price"].apply(lambda v: fmt_currency(v, currency) if pd.notna(v) else "—")
        df_sum["ChangeStr"] = df_sum["Change24h"].apply(lambda v: f"{v:+.2f}%" if pd.notna(v) else "—")
        st.dataframe(df_sum[["Symbol","PriceStr","ChangeStr"]], hide_index=True)
    else:
        st.info("No data.")

# ---------- Forecast Tab ----------
with tab_forecast:
    st.header("Forecast (Moving Avg)")
    f_sym = st.selectbox("Choose symbol", options=symbols, index=0)
    f_horizon = st.selectbox("Horizon (days)", [3,7,14,30], index=1)

    df_f = fetch_yf(f_sym, period="6mo", interval="1d")
    cs = series_close(df_f).dropna()
    if df_f.empty or cs.empty:
        st.warning("No historical data.")
    else:
        st.subheader(f"Historical: {f_sym}")
        st.line_chart(df_f.set_index("Date")[cs.name])

        arr = moving_avg_forecast(cs, f_horizon)
        last_date = df_f["Date"].iloc[-1]
        fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]
        fc_table = pd.DataFrame({"Date": fc_dates, "Predicted": np.round(arr, 4)})
        st.subheader("Forecast")
        st.table(fc_table)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_f["Date"], y=cs, mode="lines", name="History"))
        fig.add_trace(go.Scatter(x=fc_dates, y=arr, mode="lines+markers",
                                 name=f"Forecast {f_horizon}d",
                                 line=dict(dash="dash", color="#f5d76e")))
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)
