# app.py
# Forex Insight — Final Version (Market + Forecast + Signals + Gold/USD)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Forex Insight", page_icon="💹", layout="wide")

# Prophet optional
HAS_PROPHET = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except:
    HAS_PROPHET = False

# ---------------- Helpers ----------------
def ensure_flat_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

@st.cache_data(ttl=300)
def fetch_yf(symbol: str, period="6mo", interval="1d"):
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
    avg_pct = s.pct_change().dropna().mean() if s.shape[0]>1 else 0
    return np.array([ last * ((1+avg_pct)**i) for i in range(1, days+1) ])

# --- Indicators for signals ---
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

def compute_signal(close_series):
    if close_series.dropna().empty:
        return ("NO DATA","#9e9e9e")
    cs = close_series.dropna()
    ma20 = cs.rolling(20, min_periods=1).mean().iloc[-1]
    ma50 = cs.rolling(50, min_periods=1).mean().iloc[-1]
    rsi = simple_rsi(cs).iloc[-1]
    _,_,macd_diff = simple_macd(cs)
    macd_last = macd_diff.iloc[-1]

    score=0
    if ma20 > ma50: score+=1
    else: score-=1
    if rsi < 30: score+=1
    elif rsi > 70: score-=1
    if macd_last > 0: score+=1
    else: score-=1

    if score>=2: return ("BUY","#b2ff66")
    if score<=-2: return ("SELL","#ff7b7b")
    return ("HOLD","#cfd8dc")

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)

# 20 Forex pairs + Gold/USD
default_symbols = [
    "XAU=X", "EURUSD=X","GBPUSD=X","USDJPY=X","USDCHF=X","AUDUSD=X",
    "NZDUSD=X","USDCAD=X","EURGBP=X","EURJPY=X","GBPJPY=X","AUDJPY=X",
    "CHFJPY=X","NZDJPY=X","EURAUD=X","EURCAD=X","EURCHF=X","AUDCAD=X",
    "AUDNZD=X","GBPCAD=X"
]

symbols_default = st.sidebar.text_area("Symbols (comma separated)", value=",".join(default_symbols))
symbols = [s.strip().upper() for s in symbols_default.split(",") if s.strip()]

manual = st.sidebar.text_input("Add single symbol (e.g. USDTRY=X):", value="")
if manual:
    m = manual.strip().upper()
    if m not in symbols: symbols.insert(0,m)

# ---------------- Tabs ----------------
tabs = st.tabs(["Market","Forecast","About"])
tab_market, tab_forecast, tab_about = tabs

# ---------------- Market Tab ----------------
with tab_market:
    st.header("Forex Market Overview")

    summary=[]
    for s in symbols:
        d = fetch_yf(s, period=period, interval=interval)
        if d.empty or 'Close' not in d.columns:
            summary.append({"Symbol":s,"Price":None,"Change24h":None,"Signal":"NO DATA","Color":"#9e9e9e"})
            continue
        d = d.sort_values('Date').reset_index(drop=True)
        price = float(d['Close'].iloc[-1])
        prev = float(d['Close'].iloc[-2]) if d.shape[0]>=2 else price
        change24 = (price - prev)/prev*100 if prev!=0 else 0.0
        label,color = compute_signal(d['Close'])
        summary.append({"Symbol":s,"Price":price,"Change24h":round(change24,2),"Signal":label,"Color":color})

    df_sum = pd.DataFrame(summary)
    if not df_sum.empty:
        df_sum['PriceStr'] = df_sum['Price'].apply(lambda v: f"{v:,.4f}" if v is not None else "—")
        df_sum['ChangeStr'] = df_sum['Change24h'].apply(lambda v: f"{v:+.2f}%" if v is not None else "—")
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

# ---------------- Forecast Tab ----------------
with tab_forecast:
    st.header("Forecast")
    f_sym = st.selectbox("Choose symbol", options=symbols, index=0 if symbols else None)
    f_period = st.selectbox("History period", ["3mo","6mo","1y"], index=0)
    f_model = st.selectbox("Model", ["Prophet (if installed)","MovingAvg (fallback)"], index=0)
    f_horizon = st.selectbox("Horizon (days)", [3,7,14,30], index=1)

    if f_sym:
        df_f = fetch_yf(f_sym, period=f_period, interval="1d")
        if df_f.empty:
            st.warning("No historical data.")
        else:
            st.subheader(f"Historical: {f_sym}")
            st.line_chart(df_f.set_index('Date')['Close'])

            forecast_vals=None
            with st.spinner("Running forecast..."):
                try:
                    if f_model.startswith("Prophet") and HAS_PROPHET and df_f.shape[0] > 10:
                        pf = df_f[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
                        m = Prophet(daily_seasonality=True, weekly_seasonality=True)
                        m.fit(pf)
                        future = m.make_future_dataframe(periods=f_horizon, freq='D')
                        pred = m.predict(future)
                        tail = pred.tail(f_horizon)
                        forecast_vals = tail['yhat'].values
                        fc_dates = tail['ds'].dt.date.values
                    else:
                        arr = moving_avg_forecast(df_f['Close'], f_horizon)
                        forecast_vals = arr
                        last_date = df_f['Date'].iloc[-1]
                        fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]
                except Exception as e:
                    st.error("Forecast error: " + str(e))
                    arr = moving_avg_forecast(df_f['Close'], f_horizon)
                    forecast_vals = arr
                    last_date = df_f['Date'].iloc[-1]
                    fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]

            fc_table = pd.DataFrame({"Date":fc_dates, "Predicted":np.round(forecast_vals,5)})
            st.subheader("Forecast Results")
            st.table(fc_table)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_f['Date'], y=df_f['Close'], mode='lines', name='History'))
            fig.add_trace(go.Scatter(x=fc_dates, y=forecast_vals, mode='lines+markers', 
                                     name=f'Forecast {f_horizon}d', line=dict(dash='dash', color='gold')))
            fig.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- About Tab ----------------
with tab_about:
    st.header("About Forex Insight")
    st.markdown("""
    **Forex Insight** — Forex dashboard with live market data, forecasts, and signals.  
    - 20 major currency pairs included by default  
    - Gold vs USD (XAU=X) highlighted  
    - Forecast via Prophet (if installed) or Moving Average fallback  
    - Simple trading signals (MA, RSI, MACD)  
    """)
