# app.py
# Global Crypto Insight — Stable (no cache decorators)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# ---------- Page setup ----------
st.set_page_config(page_title="Global Crypto Insight", page_icon="🟡", layout="wide")

# Optional libs
HAS_PROPHET = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

HAS_TA = False
try:
    import ta
    HAS_TA = True
except Exception:
    HAS_TA = False

pd.options.mode.chained_assignment = None  # quiet pandas warnings

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
    col = "Close" if "Close" in df.columns else None
    if col is None:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors="coerce")
    return s

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

def simple_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1.0 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def simple_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff

def compute_combined_signal(close_series, next_forecast=None):
    cs = pd.to_numeric(close_series, errors="coerce").dropna()
    if cs.empty:
        return ("NO DATA", "#9e9e9e", "insufficient data")

    ma20 = cs.rolling(20, min_periods=1).mean().iloc[-1]
    ma50 = cs.rolling(50, min_periods=1).mean().iloc[-1]
    ma200 = cs.rolling(200, min_periods=1).mean().iloc[-1]

    try:
        if HAS_TA:
            rsi = ta.momentum.RSIIndicator(cs, window=14).rsi().iloc[-1]
            macd_diff = ta.trend.MACD(cs).macd_diff().iloc[-1]
        else:
            rsi = simple_rsi(cs).iloc[-1]
            _, _, md = simple_macd(cs)
            macd_diff = md.iloc[-1]
    except Exception:
        rsi = np.nan; macd_diff = np.nan

    score, reasons = 0, []
    if ma20 > ma50: score += 2; reasons.append("MA20>MA50")
    else: score -= 1; reasons.append("MA20<=MA50")
    if np.isfinite(ma200):
        if ma50 > ma200: score += 1; reasons.append("MA50>MA200")
        else: score -= 1; reasons.append("MA50<=MA200")
    if pd.notna(rsi):
        if rsi < 30: score += 1; reasons.append(f"RSI low ({rsi:.1f})")
        elif rsi > 70: score -= 1; reasons.append(f"RSI high ({rsi:.1f})")
    if pd.notna(macd_diff):
        if macd_diff > 0: score += 1; reasons.append("MACD+")
        else: score -= 1; reasons.append("MACD-")
    if next_forecast is not None and np.isfinite(next_forecast):
        last = float(cs.iloc[-1])
        if last != 0:
            pct = (next_forecast - last) / last
            if pct > 0.01: score += 1; reasons.append(f"Forecast +{pct*100:.2f}%")
            elif pct < -0.01: score -= 1; reasons.append(f"Forecast {pct*100:.2f}%")

    if score >= 4: return ("STRONG BUY", "#d4ffb3", " · ".join(reasons))
    if score >= 2: return ("BUY", "#b2ff66", " · ".join(reasons))
    if score == 1: return ("MILD BUY", "#ffe36b", " · ".join(reasons))
    if score == 0: return ("HOLD", "#cfd8dc", " · ".join(reasons))
    if score == -1: return ("MILD SELL", "#ffb86b", " · ".join(reasons))
    return ("SELL", "#ff7b7b", " · ".join(reasons))

def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        if r.status_code == 200:
            j = r.json()
            if "data" in j and len(j["data"]) > 0:
                e = j["data"][0]
                return {
                    "value": int(e.get("value", 50)),
                    "class": e.get("value_classification", "Neutral"),
                    "date": datetime.utcfromtimestamp(int(e.get("timestamp", 0))).strftime("%Y-%m-%d")
                            if e.get("timestamp") else None
                }
    except Exception:
        pass
    return {"value": None, "class": "N/A", "date": None}

def fmt_currency(x, cur="USD"):
    try:
        return f"{x:,.2f} {cur}"
    except Exception:
        return "—"

def get_fx(target):
    if target == "USD": return 1.0
    mp = {"CAD": "USDCAD=X", "EUR": "USDEUR=X", "GBP": "USDGBP=X"}
    t = mp.get(target)
    if not t: return 1.0
    try:
        df = yf.download(t, period="5d", interval="1d", progress=False)
        if df is None or df.empty: return 1.0
        df = normalize_ohlc_index(df)
        s = series_close(df).dropna()
        if s.empty: return 1.0
        return float(s.iloc[-1])
    except Exception:
        return 1.0

# ---------- Sidebar ----------
st.sidebar.header("Settings")

TOPSYMBOLS = ["BTC-USD","ETH-USD","SOL-USD","XRP-USD","ADA-USD","ENA-USD"]

currency = st.sidebar.selectbox("Display currency", ["USD","CAD","EUR","GBP"], index=0)
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)

primary = st.sidebar.selectbox("Primary symbol", TOPSYMBOLS, index=0)
extra = st.sidebar.text_area("Extra symbols (comma separated)", value="DOGE-USD,MATIC-USD")
extra_list = [s.strip().upper() for s in extra.split(",") if s.strip()]

symbols = []
for s in [primary] + [x for x in TOPSYMBOLS if x != primary] + extra_list:
    if s not in symbols:
        symbols.append(s)

fx = get_fx(currency)

# ---------- Tabs ----------
tabs = st.tabs(["Market","Forecast"])
tab_market, tab_forecast = tabs

# ---------- Market ----------
with tab_market:
    st.header("Market Overview")
    fg = fetch_fear_greed()
    st.write("Fear & Greed:", fg)

    summary = []
    for s in symbols[:10]:
        d = fetch_yf(s, period=period, interval=interval)
        c = series_close(d).dropna()
        if c.empty:
            summary.append({"Symbol": s, "Price": None, "Signal": "NO DATA"})
            continue
        price = float(c.iloc[-1]) * fx
        fc1 = moving_avg_forecast(c, 1)
        next_fc = float(fc1[0]) if len(fc1) > 0 else None
        label, _, reason = compute_combined_signal(c, next_fc)
        summary.append({"Symbol": s, "Price": price, "Signal": label, "Reason": reason})

    st.dataframe(pd.DataFrame(summary))

# ---------- Forecast ----------
with tab_forecast:
    st.header("Forecast")
    f_sym = st.selectbox("Choose symbol", options=symbols, index=0)
    f_horizon = st.slider("Forecast horizon (days)", 3, 30, 7)

    df_f = fetch_yf(f_sym, period=period, interval=interval)
    cs = series_close(df_f).dropna()
    if df_f.empty or cs.empty:
        st.warning("No historical data.")
    else:
        st.line_chart(df_f.set_index("Date")[cs.name])
        arr = moving_avg_forecast(cs, f_horizon)
        fc_dates = [(df_f["Date"].iloc[-1] + timedelta(days=i+1)).date() for i in range(f_horizon)]
        st.table(pd.DataFrame({"Date": fc_dates, "Predicted": np.round(arr, 2)}))
