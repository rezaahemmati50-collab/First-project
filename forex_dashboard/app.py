# app.py
# Forex Insight — Final (Gold + Black theme)
# Streamlit app with forecasts, signals, gold vs USD, and major forex pairs.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

st.set_page_config(page_title="Forex Insight", page_icon="💱", layout="wide")

# ---------------- Header ----------------
st.markdown(
    """
    <style>
    .fx-header{
      padding:20px;border-radius:12px;
      background: linear-gradient(90deg,#0b0b0b 0%, #1a1a1a 50%, #3a2b10 100%);
      color: #f9f4ef; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
      margin-bottom:20px;
    }
    .fx-title{font-size:28px;font-weight:800;color:#f5d76e}
    .fx-sub{color:#e6dccd;margin-top:6px}
    </style>
    <div class="fx-header">
      <div class="fx-title">Forex Insight</div>
      <div class="fx-sub">Live Forex Market · Forecasts · Buy/Sell Signals · Gold vs USD</div>
    </div>
    """, unsafe_allow_html=True
)

# ---------------- Optional libs ----------------
HAS_PROPHET = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except:
    pass

# ---------------- Helpers ----------------
def ensure_flat_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

@st.cache_data(ttl=300)
def fetch_yf(symbol, period="6mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty: return pd.DataFrame()
        df = ensure_flat_columns(df).reset_index()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except:
        return pd.DataFrame()

def moving_avg_forecast(series, days):
    s = series.dropna()
    if s.empty: return np.array([np.nan]*days)
    last = float(s.iloc[-1])
    avg_pct = s.pct_change().dropna().mean() if s.shape[0]>2 else 0.0
    return np.array([ last*((1+avg_pct)**i) for i in range(1,days+1) ])

def simple_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up/(ma_down+1e-10)
    return 100-(100/(1+rs))

def simple_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast-ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal, macd-macd_signal

def compute_signal(series, next_fc=None):
    if series.dropna().empty: return ("NO DATA","#9e9e9e","")
    cs = series.dropna()
    ma20,ma50,ma200 = cs.rolling(20).mean().iloc[-1], cs.rolling(50).mean().iloc[-1], cs.rolling(200).mean().iloc[-1]
    rsi = simple_rsi(cs).iloc[-1]
    _,_,macd_diff = simple_macd(cs); macd = macd_diff.iloc[-1]

    score=0; reasons=[]
    if ma20>ma50: score+=1; reasons.append("MA20>MA50")
    else: score-=1; reasons.append("MA20<=MA50")
    if ma50>ma200: score+=1; reasons.append("MA50>MA200")
    else: score-=1; reasons.append("MA50<=MA200")
    if rsi<30: score+=1; reasons.append("RSI low")
    elif rsi>70: score-=1; reasons.append("RSI high")
    if macd>0: score+=1; reasons.append("MACD +")
    else: score-=1; reasons.append("MACD -")
    if next_fc is not None:
        last=float(cs.iloc[-1]); pct=(next_fc-last)/last
        if pct>0.01: score+=1; reasons.append(f"Forecast +{pct*100:.1f}%")
        elif pct<-0.01: score-=1; reasons.append(f"Forecast {pct*100:.1f}%")

    if score>=3: return ("BUY","#b2ff66"," · ".join(reasons))
    if score>=1: return ("HOLD","#ffe36b"," · ".join(reasons))
    return ("SELL","#ff7b7b"," · ".join(reasons))

def fmt_cur(x): 
    return f"{x:,.4f}" if x is not None else "—"

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
period = st.sidebar.selectbox("History period", ["3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)
horizon = st.sidebar.selectbox("Forecast horizon (days)", [3,7,14,30], index=1)

default_symbols = [
    "XAUUSD=X","GC=F", # Gold Spot + Futures
    "EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X","AUDUSD=X",
    "NZDUSD=X","USDCAD=X","EURGBP=X","EURJPY=X","GBPJPY=X",
    "AUDJPY=X","CHFJPY=X","NZDJPY=X","EURAUD=X","EURCAD=X",
    "EURCHF=X","AUDCAD=X","AUDNZD=X","GBPCAD=X"
]

symbols_input = st.sidebar.text_area("Symbols", value=",".join(default_symbols))
symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

manual = st.sidebar.text_input("Add single symbol (e.g. USDTRY=X):", value="")
if manual and manual not in symbols:
    symbols.insert(0, manual.strip().upper())

# ---------------- Tabs ----------------
tabs = st.tabs(["Market","Forecast","About"])
tab_market, tab_forecast, tab_about = tabs

# ---------------- Market Tab ----------------
with tab_market:
    st.header("Market Overview")
    rows=[]
    for s in symbols:
        d=fetch_yf(s,period=period,interval=interval)
        if d.empty or "Close" not in d.columns:
            rows.append({"Symbol":s,"Price":None,"Change":None,"Signal":"NO DATA","Color":"#9e9e9e"})
            continue
        d=d.sort_values("Date")
        price=float(d["Close"].iloc[-1])
        prev=float(d["Close"].iloc[-2]) if len(d)>1 else price
        change=(price-prev)/prev*100 if prev!=0 else 0
        fc=moving_avg_forecast(d["Close"],1)
        next_fc=fc[0] if len(fc)>0 else None
        sig,color,reason=compute_signal(d["Close"],next_fc)
        rows.append({"Symbol":s,"Price":price,"Change":change,"Signal":sig,"Color":color})
    df=pd.DataFrame(rows)
    if not df.empty:
        df["PriceStr"]=df["Price"].apply(fmt_cur)
        df["ChangeStr"]=df["Change"].apply(lambda v: f"{v:+.2f}%" if v is not None else "—")
        st.dataframe(df[["Symbol","PriceStr","ChangeStr","Signal"]], use_container_width=True)

        st.subheader("Signal Cards")
        cols=st.columns(min(6,len(df)))
        for i,row in df.iterrows():
            c=cols[i%len(cols)]
            with c:
                st.markdown(f"<div style='background:{row['Color']};padding:10px;border-radius:8px;text-align:center;'><b>{row['Symbol']}</b><br/>{row['Signal']}<br/>{row['PriceStr']} · {row['ChangeStr']}</div>", unsafe_allow_html=True)
    else:
        st.info("No data.")

# ---------------- Forecast Tab ----------------
with tab_forecast:
    st.header("Forecast")
    f_sym=st.selectbox("Choose symbol", options=symbols)
    if f_sym:
        df_f=fetch_yf(f_sym,period=period,interval=interval)
        if df_f.empty:
            st.warning("No historical data.")
        else:
            st.line_chart(df_f.set_index("Date")["Close"])
            with st.spinner("Running forecast..."):
                if HAS_PROPHET and len(df_f)>30:
                    pf=df_f[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"})
                    m=Prophet(daily_seasonality=True,weekly_seasonality=True)
                    m.fit(pf)
                    future=m.make_future_dataframe(periods=horizon)
                    fcst=m.predict(future).tail(horizon)
                    st.write(fcst[["ds","yhat"]])
                else:
                    arr=moving_avg_forecast(df_f["Close"],horizon)
                    last_date=df_f["Date"].iloc[-1]
                    fc_dates=[last_date+timedelta(days=i+1) for i in range(horizon)]
                    st.table(pd.DataFrame({"Date":fc_dates,"Forecast":arr}))

# ---------------- About ----------------
with tab_about:
    st.header("About")
    st.markdown("**Forex Insight** — Dashboard for FX majors + Gold. Forecasts, signals, heatmap. Educational only.")
