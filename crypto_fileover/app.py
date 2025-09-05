import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Dashboard (Binance)", page_icon="🟡", layout="wide")

def fetch_binance(symbol="BTCUSDT", interval="1d", limit=200):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        rows=[]
        for k in data:
            rows.append({
                "Date": datetime.fromtimestamp(k[0]/1000),
                "Open": float(k[1]),
                "High": float(k[2]),
                "Low": float(k[3]),
                "Close": float(k[4]),
                "Volume": float(k[5])
            })
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Binance error: {e}")
        return pd.DataFrame()

st.sidebar.header("Settings")
symbols = st.sidebar.text_area("Symbols (comma separated)", value="BTCUSDT,ETHUSDT,ADAUSDT").split(",")

tab1,tab2 = st.tabs(["Market","Forecast"])

with tab1:
    st.header("Market Overview")
    summary=[]
    for sym in symbols:
        df = fetch_binance(sym.strip())
        if df.empty:
            summary.append({"Symbol":sym,"Price":"—"})
        else:
            last = df["Close"].iloc[-1]
            prev = df["Close"].iloc[-2] if len(df)>1 else last
            change = (last-prev)/prev*100 if prev!=0 else 0
            summary.append({"Symbol":sym,"Price":f"{last:,.2f} USD","Change":f"{change:+.2f}%"})
    st.dataframe(pd.DataFrame(summary))

with tab2:
    st.header("Simple Forecast (7 days)")
    f_sym = st.selectbox("Choose symbol", symbols, index=0)
    df = fetch_binance(f_sym.strip())
    if not df.empty:
        st.line_chart(df.set_index("Date")["Close"])
        last = df["Close"].iloc[-1]
        avg_pct = df["Close"].pct_change().dropna().mean()
        fc = [last*((1+avg_pct)**i) for i in range(1,8)]
        fc_dates = [(df["Date"].iloc[-1]+timedelta(days=i)).date() for i in range(1,8)]
        st.table(pd.DataFrame({"Date":fc_dates,"Forecast":np.round(fc,2)}))
    else:
        st.warning("No data")
