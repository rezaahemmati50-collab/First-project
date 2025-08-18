# app.py
# Global FX Insight — Final (gold + black theme)
# Streamlit dashboard for Forex pairs with forecasting & signals

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Global FX Insight", page_icon="💹", layout="wide")

# ---------------- Helpers ----------------
def ensure_flat_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

@st.cache_data(ttl=180)
def fetch_yf(symbol: str, period="3mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = ensure_flat_columns(df)
        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            df.rename(columns={df.columns[0]:'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()

def moving_avg_forecast(series, days):
    try:
        s = series.dropna()
        if s.empty:
            return np.array([np.nan]*days)
        last = float(s.iloc[-1])
        avg_pct = s.pct_change().dropna().mean() if s.pct_change().dropna().shape[0] > 0 else 0.0
        return np.array([ last * ((1+avg_pct)**i) for i in range(1, days+1) ])
    except Exception:
        return np.array([np.nan]*days)

def simple_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_combined_signal(close_series, next_forecast=None):
    if close_series.dropna().empty:
        return ("NO DATA","#9e9e9e","insufficient data")
    cs = close_series.dropna()
    ma20 = cs.rolling(20, min_periods=1).mean().iloc[-1]
    ma50 = cs.rolling(50, min_periods=1).mean().iloc[-1]

    try:
        rsi = simple_rsi(cs).iloc[-1]
    except Exception:
        rsi = np.nan

    score = 0; reasons = []
    if ma20 > ma50:
        score += 1; reasons.append("MA20>MA50")
    else:
        score -= 1; reasons.append("MA20<=MA50")

    if not np.isnan(rsi):
        if rsi < 30:
            score += 1; reasons.append(f"RSI low ({rsi:.1f})")
        elif rsi > 70:
            score -= 1; reasons.append(f"RSI high ({rsi:.1f})")

    if next_forecast is not None and not np.isnan(next_forecast):
        last = float(cs.iloc[-1])
        pct = (next_forecast - last) / last
        if pct > 0.01:
            score += 1; reasons.append(f"Forecast +{pct*100:.2f}%")
        elif pct < -0.01:
            score -= 1; reasons.append(f"Forecast {pct*100:.2f}%")

    if score >= 2: return ("BUY","#b2ff66"," · ".join(reasons))
    if score == 1: return ("MILD BUY","#ffe36b"," · ".join(reasons))
    if score == 0: return ("HOLD","#cfd8dc"," · ".join(reasons))
    if score == -1: return ("MILD SELL","#ffb86b"," · ".join(reasons))
    return ("SELL","#ff7b7b"," · ".join(reasons))

def fmt_currency(x):
    try:
        return f"{x:,.4f}"
    except Exception:
        return "—"

# ---------------- Header ----------------
st.markdown(
    """
    <style>
    .gci-header{
      padding:20px;border-radius:12px;
      background: linear-gradient(90deg,#0b0b0b 0%, #1a1a1a 50%, #3a2b10 100%);
      color: #f9f4ef; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .gci-title{font-size:26px;font-weight:800;color:#f5d76e}
    .gci-sub{color:#e6dccd;margin-top:6px}
    </style>
    <div class="gci-header">
      <div class="gci-title">Global FX Insight</div>
      <div class="gci-sub">Live Forex · Forecasts · Signals · Heatmap</div>
    </div>
    """, unsafe_allow_html=True
)

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)

st.sidebar.markdown("---")
symbols_default = st.sidebar.text_area(
    "FX Pairs (comma separated)",
    value="XAUUSD=X,EURUSD=X,USDJPY=X,GBPUSD=X,USDCHF=X,AUDUSD=X,USDCAD=X,"
          "NZDUSD=X,EURJPY=X,GBPJPY=X,EURGBP=X,AUDJPY=X,NZDJPY=X,EURCHF=X,"
          "CADJPY=X,CHFJPY=X,EURAUD=X,GBPAUD=X,GBPCAD=X,GBPNZD=X,AUDNZD=X"
).upper()
symbols = [s.strip() for s in symbols_default.split(",") if s.strip()]

st.sidebar.markdown("---")
manual = st.sidebar.text_input("Add single pair (e.g. EURUSD=X):", value="")
if manual:
    m = manual.strip().upper()
    if m not in symbols:
        symbols.insert(0,m)

# ---------------- Market ----------------
st.header("Market Overview")

summary=[]
with st.spinner("Fetching FX data..."):
    for s in symbols:
        d = fetch_yf(s, period=period, interval=interval)
        if d.empty or 'Close' not in d.columns:
            summary.append({"Symbol":s,"Price":None,"Change24h":None,"Signal":"NO DATA","Color":"#9e9e9e"})
            continue
        d = d.sort_values('Date').reset_index(drop=True)
        price = float(d['Close'].iloc[-1])
        prev = float(d['Close'].iloc[-2]) if d['Close'].dropna().shape[0]>=2 else price
        change24 = (price - prev)/prev*100 if prev!=0 else 0.0
        fc = moving_avg_forecast(d['Close'],1)
        next_fc = float(fc[0]) if len(fc)>0 else None
        label,color,reason = compute_combined_signal(d['Close'], next_fc)
        summary.append({"Symbol":s,"Price":price,"Change24h":round(change24,2),"Signal":label,"Color":color,"Reason":reason})
df_sum = pd.DataFrame(summary)

if not df_sum.empty:
    df_sum['PriceStr'] = df_sum['Price'].apply(lambda v: fmt_currency(v) if v is not None else "—")
    df_sum['ChangeStr'] = df_sum['Change24h'].apply(lambda v: f"{v:+.2f}%")
    st.dataframe(df_sum[['Symbol','PriceStr','ChangeStr','Signal']], use_container_width=True)

    st.markdown("### Signal Cards")
    cols = st.columns(min(6, max(1,len(df_sum))))
    for i,row in df_sum.iterrows():
        c = cols[i % len(cols)]
        with c:
            html = f"<div style='background:{row['Color']};padding:10px;border-radius:8px;text-align:center;color:#021014;'><strong>{row['Symbol']}</strong><br/>{row['Signal']}<br/>{row['PriceStr']} · {row['ChangeStr']}</div>"
            st.markdown(html, unsafe_allow_html=True)
else:
    st.info("No data available.")
