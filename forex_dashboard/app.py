# app.py
# Global Forex Insight — polished forex dashboard
# Based on Global Crypto Insight but adapted for FX pairs

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="Global Forex Insight", page_icon="💱", layout="wide")

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

def simple_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff

def compute_combined_signal(close_series, next_forecast=None):
    if close_series.dropna().empty:
        return ("NO DATA","#9e9e9e","insufficient data")
    cs = close_series.dropna()
    ma20 = cs.rolling(20, min_periods=1).mean().iloc[-1]
    ma50 = cs.rolling(50, min_periods=1).mean().iloc[-1]
    ma200 = cs.rolling(200, min_periods=1).mean().iloc[-1]

    rsi = simple_rsi(cs).iloc[-1]
    _,_,macd_diff_series = simple_macd(cs)
    macd_diff = macd_diff_series.iloc[-1]

    score = 0; reasons = []
    if ma20 > ma50: score += 2; reasons.append("MA20>MA50")
    else: score -= 1; reasons.append("MA20<=MA50")

    if ma50 > ma200: score += 1; reasons.append("MA50>MA200")
    else: score -= 1; reasons.append("MA50<=MA200")

    if rsi < 30: score += 1; reasons.append(f"RSI low ({rsi:.1f})")
    elif rsi > 70: score -= 1; reasons.append(f"RSI high ({rsi:.1f})")

    if macd_diff > 0: score += 1; reasons.append("MACD positive")
    else: score -= 1; reasons.append("MACD negative")

    if next_forecast is not None and not np.isnan(next_forecast):
        last = float(cs.iloc[-1])
        pct = (next_forecast - last) / last
        if pct > 0.001: score += 1; reasons.append(f"Forecast +{pct*100:.2f}%")
        elif pct < -0.001: score -= 1; reasons.append(f"Forecast {pct*100:.2f}%")

    if score >= 4: return ("STRONG BUY","#d4ffb3"," · ".join(reasons))
    if score >= 2: return ("BUY","#b2ff66"," · ".join(reasons))
    if score == 1: return ("MILD BUY","#ffe36b"," · ".join(reasons))
    if score == 0: return ("HOLD","#cfd8dc"," · ".join(reasons))
    if score == -1: return ("MILD SELL","#ffb86b"," · ".join(reasons))
    return ("SELL","#ff7b7b"," · ".join(reasons))

def fmt_currency(x, cur="USD"):
    try:
        return f"{x:,.4f} {cur}"
    except Exception:
        return "—"

# ---------------- Header ----------------
st.markdown(
    """
    <style>
    .gfi-header{
      padding:20px;border-radius:12px;
      background: linear-gradient(90deg,#0b0b0b 0%, #1a1a1a 50%, #102a3a 100%);
      color: #f9f4ef; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .gfi-title{font-size:26px;font-weight:800;color:#4ac0ff}
    .gfi-sub{color:#e6dccd;margin-top:6px}
    </style>
    <div class="gfi-header">
      <div class="gfi-title">Global Forex Insight</div>
      <div class="gfi-sub">Major FX pairs · Multi-model forecasts · Signals · Heatmap</div>
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
    value="EURUSD=X,GBPUSD=X,USDJPY=X,USDCHF=X,AUDUSD=X,USDCAD=X,NZDUSD=X"
).upper()
symbols = [s.strip() for s in symbols_default.split(",") if s.strip()]

st.sidebar.markdown("---")
manual = st.sidebar.text_input("Add single pair (e.g. EURUSD=X):", value="")
if manual:
    m = manual.strip().upper()
    if m not in symbols:
        symbols.insert(0,m)

# Tabs
tabs = st.tabs(["Market","Forecast","About"])
tab_market, tab_forecast, tab_about = tabs

# ---------------- Market Tab ----------------
with tab_market:
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
        df_sum['PriceStr'] = df_sum['Price'].apply(lambda v: fmt_currency(v,""))
        df_sum['ChangeStr'] = df_sum['Change24h'].apply(lambda v: f"{v:+.2f}%")
        st.dataframe(df_sum[['Symbol','PriceStr','ChangeStr','Signal']], use_container_width=True)
    else:
        st.info("No data.")

    st.markdown("### Signal Cards")
    cols = st.columns(min(6, max(1,len(df_sum))))
    for i,row in df_sum.iterrows():
        c = cols[i % len(cols)]
        with c:
            html = f"<div style='background:{row['Color']};padding:10px;border-radius:8px;text-align:center;color:#021014;'><strong>{row['Symbol']}</strong><br/>{row['Signal']}<br/>{row['PriceStr']} · {row['ChangeStr']}</div>"
            st.markdown(html, unsafe_allow_html=True)

# ---------------- Forecast Tab ----------------
with tab_forecast:
    st.header("Forecast")
    f_sym = st.selectbox("Choose FX pair", options=symbols, index=0 if symbols else None)
    f_period = st.selectbox("History period", ["3mo","6mo","1y"], index=0)
    f_interval = st.selectbox("Interval", ["1d","1h"], index=0)
    f_horizon = st.selectbox("Horizon (days)", [3,7,14], index=1)

    if f_sym:
        df_f = fetch_yf(f_sym, period=f_period, interval=f_interval)
        if df_f.empty:
            st.warning("No historical data.")
        else:
            st.subheader(f"Historical: {f_sym}")
            st.line_chart(df_f.set_index('Date')['Close'])
            arr = moving_avg_forecast(df_f['Close'], f_horizon)
            last_date = df_f['Date'].iloc[-1]
            fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]
            st.subheader("Forecast")
            st.table(pd.DataFrame({"Date":fc_dates,"Predicted":np.round(arr,4)}))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_f['Date'], y=df_f['Close'], mode='lines', name='History'))
            fig.add_trace(go.Scatter(x=fc_dates, y=arr, mode='lines+markers', name=f'Forecast {f_horizon}d', line=dict(dash='dash', color='#4ac0ff')))
            fig.update_layout(template='plotly_dark', height=520)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- About ----------------
with tab_about:
    st.header("About")
    st.markdown("""
    **Global Forex Insight** — dashboard for FX traders.
    - Live prices for major currency pairs
    - Technical signals (MA/RSI/MACD)
    - Moving average forecast
    """)
    st.caption("Educational only — not financial advice.")
