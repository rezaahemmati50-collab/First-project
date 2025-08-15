# app.py
# Global Crypto Insight — Dual-signal Mode (Original + Market-filtered)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Global Crypto Insight", page_icon="🟡", layout="wide")

# Optional libs
HAS_PROPHET, HAS_TA = False, False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except:
    pass
try:
    import ta
    HAS_TA = True
except:
    pass

pd.options.mode.chained_assignment = None

# Helpers
def ensure_flat_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def normalize_ohlc_index(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = ensure_flat_columns(df).reset_index()
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    expected = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    return df[["Date"] + expected].dropna(subset=["Date"]).sort_values("Date")

@st.cache_data(ttl=180)
def fetch_yf(symbol, period="3mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return normalize_ohlc_index(df)
    except:
        return pd.DataFrame()

def series_close(df):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "Close" in df.columns:
        col = "Close"
    else:
        col = None
    if col is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")

def moving_avg_forecast(close_s, days):
    try:
        s = pd.to_numeric(close_s, errors="coerce").dropna()
        if s.empty:
            return np.array([np.nan]*days)
        last = float(s.iloc[-1])
        avg_pct = s.pct_change().dropna().mean() if s.shape[0] > 1 else 0.0
        return np.array([last * ((1 + (avg_pct if np.isfinite(avg_pct) else 0.0)) ** i) for i in range(1, days+1)])
    except:
        return np.array([np.nan]*days)

def simple_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1.0 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def simple_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal, macd - macd_signal

def compute_signal(close_series, next_forecast=None, mode="original"):
    cs = pd.to_numeric(close_series, errors="coerce").dropna()
    if cs.empty:
        return ("NO DATA", "#9e9e9e", "no data")
    ma20 = cs.rolling(20, min_periods=1).mean().iloc[-1]
    ma50 = cs.rolling(50, min_periods=1).mean().iloc[-1]
    ma200 = cs.rolling(200, min_periods=1).mean().iloc[-1]

    try:
        if HAS_TA:
            rsi = ta.momentum.RSIIndicator(cs, 14).rsi().iloc[-1]
            macd_diff = ta.trend.MACD(cs).macd_diff().iloc[-1]
        else:
            rsi = simple_rsi(cs).iloc[-1]
            _, _, md = simple_macd(cs)
            macd_diff = md.iloc[-1]
    except:
        rsi, macd_diff = np.nan, np.nan

    score, reasons = 0, []

    # MAs
    if ma20 > ma50:
        score += 2; reasons.append("MA20>MA50")
    else:
        score -= 1; reasons.append("MA20<=MA50")
    if np.isfinite(ma200):
        if ma50 > ma200:
            score += 1; reasons.append("MA50>MA200")
        else:
            score -= 1; reasons.append("MA50<=MA200")

    # RSI
    if pd.notna(rsi):
        if rsi < 30:
            score += 1; reasons.append(f"RSI low ({rsi:.1f})")
        elif rsi > 70:
            score -= 1; reasons.append(f"RSI high ({rsi:.1f})")

    # MACD
    if pd.notna(macd_diff):
        if macd_diff > 0:
            score += 1; reasons.append("MACD+")
        else:
            score -= 1; reasons.append("MACD-")

    # Forecast tilt
    if next_forecast is not None and np.isfinite(next_forecast):
        last = float(cs.iloc[-1])
        if last != 0:
            pct = (next_forecast - last) / last
            if pct > 0.01:
                score += 1; reasons.append(f"Forecast +{pct*100:.2f}%")
            elif pct < -0.01:
                score -= 1; reasons.append(f"Forecast {pct*100:.2f}%")

    # Market filter mode
    if mode == "filtered":
        if rsi > 60 and ma20 < ma50:
            score -= 2; reasons.append("Market filter: downtrend with overbought RSI")

    if score >= 4: return ("STRONG BUY", "#d4ffb3", " · ".join(reasons))
    if score >= 2: return ("BUY", "#b2ff66", " · ".join(reasons))
    if score == 1: return ("MILD BUY", "#ffe36b", " · ".join(reasons))
    if score == 0: return ("HOLD", "#cfd8dc", " · ".join(reasons))
    if score == -1: return ("MILD SELL", "#ffb86b", " · ".join(reasons))
    return ("SELL", "#ff7b7b", " · ".join(reasons))

@st.cache_data(ttl=600)
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
                }
    except:
        pass
    return {"value": None, "class": "N/A", "date": None}

# ---------------- UI ----------------
st.sidebar.header("Settings")

# Updated coin list
COINS = [
    "BTC-USD","ETH-USD","USDT-USD","BNB-USD","SOL-USD",
    "XRP-USD","USDC-USD","DOGE-USD","TON-USD","ADA-USD",
    "TRX-USD","AVAX-USD","SHIB-USD","DOT-USD","BCH-USD",
    "LINK-USD","NEAR-USD","LTC-USD","DAI-USD","ENA-USD"
]

currency = st.sidebar.selectbox("Currency", ["USD","CAD","EUR","GBP"], index=0)
period = st.sidebar.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)
signal_mode = st.sidebar.radio("Signal Mode", ["Original","Filtered"], index=0)

primary = st.sidebar.selectbox("Primary", COINS, index=0)

# ---------- Market Tab ----------
st.header("Market Overview")
summary = []
for s in COINS:
    d = fetch_yf(s, period=period, interval=interval)
    c = series_close(d).dropna()
    if c.empty:
        summary.append({"Symbol": s, "Signal": "NO DATA", "Color": "#9e9e9e"})
        continue
    fc1 = moving_avg_forecast(c, 1)
    next_fc = float(fc1[0]) if len(fc1) > 0 else None
    label, color, reason = compute_signal(c, next_fc, mode="filtered" if signal_mode=="Filtered" else "original")
    summary.append({"Symbol": s, "Signal": label, "Color": color})

df_sum = pd.DataFrame(summary)
st.dataframe(df_sum, use_container_width=True)
