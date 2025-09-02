# app.py — Global Crypto Insight (Crypto with Failover Yahoo -> AlphaVantage)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Global Crypto Insight", page_icon="🟡", layout="wide")

# ---------- Failover fetcher ----------
def fetch_crypto(symbol, period="3mo", interval="1d"):
    """Try Yahoo Finance, fallback to Alpha Vantage or Binance"""
    # --- Try Yahoo Finance first ---
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is not None and not df.empty:
            df = df.reset_index()
            return df, "Yahoo Finance"
    except Exception:
        pass

    # --- Try Alpha Vantage (daily crypto) ---
    try:
        API_KEY = "demo"  # ⬅️ جایگزین با API key رایگان خودت از https://www.alphavantage.co
        if "-USD" in symbol:
            sym = symbol.replace("-USD","")
            url = f"https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol={sym}&market=USD&apikey={API_KEY}"
            r = requests.get(url, timeout=10)
            j = r.json()
            if "Time Series (Digital Currency Daily)" in j:
                data = j["Time Series (Digital Currency Daily)"]
                df = pd.DataFrame(data).T
                df.index = pd.to_datetime(df.index)
                df = df.rename(columns={"4a. close (USD)":"Close"})
                df = df[["Close"]].astype(float).reset_index().rename(columns={"index":"Date"})
                return df, "Alpha Vantage"
    except Exception:
        pass

    # --- Try Binance (spot klines) ---
    try:
        sym = symbol.replace("-USD","USDT")  # BTC-USD → BTCUSDT
        url = f"https://api.binance.com/api/v3/klines?symbol={sym}&interval=1d&limit=200"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            rows = []
            for k in data:
                ts = datetime.fromtimestamp(k[0]/1000)
                close = float(k[4])
                rows.append({"Date": ts, "Close": close})
            df = pd.DataFrame(rows)
            return df, "Binance"
    except Exception:
        pass

    return pd.DataFrame(), "No Data"

# ---------- Sidebar ----------
st.sidebar.header("Settings")
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d"], index=0)
symbols_default = st.sidebar.text_area("Symbols", value="BTC-USD,ETH-USD,ADA-USD,SOL-USD,XRP-USD").upper()
symbols = [s.strip() for s in symbols_default.split(",") if s.strip()]

# ---------- Main Tabs ----------
tabs = st.tabs(["Market","Forecast","Portfolio","About"])
tab_market, tab_forecast, tab_portfolio, tab_about = tabs

# ---------- Market ----------
with tab_market:
    st.header("Market Overview")
    summary=[]
    for s in symbols:
        d,src = fetch_crypto(s, period=period, interval=interval)
        if d.empty:
            summary.append({"Symbol":s,"Price":"—","Change":"—","Source":src})
            continue
        d = d.sort_values("Date")
        last = d["Close"].iloc[-1]
        prev = d["Close"].iloc[-2] if len(d)>1 else last
        ch = (last-prev)/prev*100 if prev!=0 else 0
        summary.append({"Symbol":s,"Price":f"{last:,.2f} USD","Change":f"{ch:+.2f}%","Source":src})
    df_sum = pd.DataFrame(summary)
    st.dataframe(df_sum, use_container_width=True)

# ---------- Forecast ----------
with tab_forecast:
    st.header("Simple Forecast")
    f_sym = st.selectbox("Choose symbol", symbols, index=0)
    df, src = fetch_crypto(f_sym, period="6mo", interval="1d")
    if df.empty:
        st.warning(f"No data for {f_sym}")
    else:
        st.line_chart(df.set_index("Date")["Close"])
        # simple moving avg forecast
        last = df["Close"].iloc[-1]
        avg_pct = df["Close"].pct_change().dropna().mean()
        horizon = 7
        fc = [last * ((1+avg_pct)**i) for i in range(1,horizon+1)]
        fc_dates = [(df["Date"].iloc[-1] + timedelta(days=i)).date() for i in range(1,horizon+1)]
        st.subheader(f"7-day forecast ({src})")
        st.table(pd.DataFrame({"Date":fc_dates,"Forecast":np.round(fc,2)}))

# ---------- Portfolio ----------
with tab_portfolio:
    st.header("Portfolio (demo)")
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = []
    s = st.text_input("Symbol (e.g. BTC-USD)")
    q = st.number_input("Quantity",0.0)
    if st.button("Add"):
        if s: st.session_state['portfolio'].append({"symbol":s,"qty":q})
    st.write(st.session_state['portfolio'])

# ---------- About ----------
with tab_about:
    st.header("About")
    st.write("This is the Crypto version with automatic failover (Yahoo → Alpha Vantage → Binance).")
    st.caption("Educational only — not financial advice.")
