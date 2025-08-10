# app.py
# Global Crypto Insight — Final upgraded
# - RSS news (CoinDesk fallback)
# - Top Movers (CoinGecko)
# - Fear & Greed (alternative.me)
# - Auto-refresh via cached TTL (default 300s)
# - Forecast: Prophet (if available) + LSTM (if TensorFlow available), fallback to moving-average
# - Robust error handling and graceful fallbacks

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta

st.set_page_config(page_title="Global Crypto Insight", page_icon="🟡", layout="wide")

# ---------------- Optional heavy libs ----------------
HAS_PROPHET = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.preprocessing import MinMaxScaler
    HAS_TF = True
except Exception:
    HAS_TF = False

HAS_TA = False
try:
    import ta
    HAS_TA = True
except Exception:
    HAS_TA = False

# ---------------- Settings ----------------
AUTO_REFRESH_SECONDS = 300  # 5 minutes default (you can change)
COINGECKO_PER_PAGE = 250    # number of coins fetched from CoinGecko for movers
NEWS_MAX = 10               # number of rss headlines to show

# ---------------- Helpers ----------------
def ensure_flat_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

@st.cache_data(ttl=180)
def fetch_yf(symbol: str, period="6mo", interval="1d"):
    """Get yfinance data and normalize Date column."""
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
    """Simple multiplicative average-percent forecast."""
    try:
        s = series.dropna().astype(float)
        if s.empty:
            return np.array([np.nan]*days)
        last = float(s.iloc[-1])
        pct = s.pct_change().dropna()
        avg_pct = float(pct.mean()) if not pct.empty else 0.0
        return np.array([ last * ((1+avg_pct)**i) for i in range(1, days+1) ])
    except Exception:
        return np.array([np.nan]*days)

def fmt_currency(x, cur="USD"):
    try:
        return f"{x:,.2f} {cur}"
    except Exception:
        return "—"

# ---------------- Fear & Greed ----------------
@st.cache_data(ttl=600)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        if r.status_code == 200:
            j = r.json()
            if "data" in j and len(j["data"])>0:
                e = j["data"][0]
                return {"value": int(e.get("value",50)),
                        "class": e.get("value_classification","Neutral"),
                        "date": datetime.utcfromtimestamp(int(e.get("timestamp",0))).strftime("%Y-%m-%d") if e.get("timestamp") else None}
    except Exception:
        pass
    return {"value": None, "class":"N/A", "date":None}

# ---------------- CoinGecko Top Movers ----------------
@st.cache_data(ttl=150)
def fetch_top_movers_coingecko(vs_currency='usd', per_page=250):
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": vs_currency,
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h"
    }
    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
        rows=[]
        for c in data:
            change = c.get("price_change_percentage_24h")
            rows.append({
                "id": c.get("id"),
                "symbol": c.get("symbol").upper(),
                "name": c.get("name"),
                "price": c.get("current_price"),
                "change24": change if change is not None else 0.0
            })
        df = pd.DataFrame(rows).dropna(subset=['change24'])
        df_sorted = df.sort_values('change24', ascending=False)
        gainers = df_sorted.head(10).head(5).to_dict('records') if not df_sorted.empty else []
        losers = df_sorted.tail(10).tail(5).to_dict('records') if not df_sorted.empty else []
        return {"gainers":gainers, "losers":losers}
    except Exception:
        return None

# ---------------- News (CoinDesk RSS fallback) ----------------
@st.cache_data(ttl=300)
def fetch_news_rss(max_items=NEWS_MAX):
    try:
        url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            txt = r.text
            parts = txt.split("<item>")
            titles = []
            for p in parts[1: max_items+1]:
                if "<title>" in p:
                    t = p.split("<title>")[1].split("</title>")[0].strip()
                    titles.append(t)
            return titles
    except Exception:
        pass
    return []

# ---------------- Simple indicators & signal ----------------
def simple_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_signal_from_close(close_series, next_fc=None):
    if close_series.dropna().empty:
        return ("NO DATA", "#9e9e9e", "no data")
    s = close_series.dropna().astype(float)
    ma20 = s.rolling(20, min_periods=1).mean().iloc[-1]
    ma50 = s.rolling(50, min_periods=1).mean().iloc[-1]
    try:
        rsi = simple_rsi(s).iloc[-1]
    except Exception:
        rsi = np.nan
    score = 0; reasons=[]
    if ma20 > ma50:
        score += 2; reasons.append("MA20>MA50")
    else:
        score -= 1; reasons.append("MA20<=MA50")
    if not np.isnan(rsi):
        if rsi < 30:
            score += 1; reasons.append(f"RSI low ({rsi:.1f})")
        elif rsi > 70:
            score -= 1; reasons.append(f"RSI high ({rsi:.1f})")
    try:
        if next_fc is not None and not np.isnan(next_fc):
            last = float(s.iloc[-1])
            pct = (next_fc - last) / last
            if pct > 0.01:
                score += 1; reasons.append(f"Forecast +{pct*100:.2f}%")
            elif pct < -0.01:
                score -= 1; reasons.append(f"Forecast {pct*100:.2f}%")
    except Exception:
        pass
    if score >= 3: return ("STRONG BUY","#b2ff66"," · ".join(reasons))
    if score >= 1: return ("BUY","#ffe36b"," · ".join(reasons))
    if score == 0: return ("HOLD","#cfd8dc"," · ".join(reasons))
    return ("SELL","#ff7b7b"," · ".join(reasons))

# ---------------- LSTM Forecast (if TF available) ----------------
@st.cache_data(ttl=600)
def forecast_lstm(series, days=7, epochs=20, units=32):
    """
    Train a tiny LSTM on the close series and forecast 'days' ahead.
    This is intentionally minimal to run reasonably; for production tune hyperparams.
    Returns numpy array of predictions length days or raises.
    """
    if not HAS_TF:
        raise RuntimeError("TensorFlow not available for LSTM forecast.")
    s = series.dropna().astype(float)
    if s.shape[0] < 30:
        raise RuntimeError("Not enough data for LSTM (need >=30).")
    # scale
    scaler = MinMaxScaler(feature_range=(0,1))
    vals = s.values.reshape(-1,1)
    scaled = scaler.fit_transform(vals)
    # prepare supervised windows
    lookback = 14
    X=[]; y=[]
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i,0])
        y.append(scaled[i,0])
    X=np.array(X); y=np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    # build model
    model = Sequential()
    model.add(LSTM(units, input_shape=(X.shape[1],1)))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # train briefly
    model.fit(X,y, epochs=epochs, batch_size=16, verbose=0)
    # forecast iterative
    last_window = scaled[-lookback:,0].reshape(1,lookback,1)
    preds=[]
    cur = last_window.copy()
    for i in range(days):
        p = model.predict(cur, verbose=0)[0,0]
        preds.append(p)
        # shift
        cur = np.roll(cur, -1, axis=1)
        cur[0,-1,0] = p
    preds = np.array(preds).reshape(-1,1)
    inv = scaler.inverse_transform(preds).reshape(-1)
    return inv

# ---------------- UI: header & sidebar ----------------
st.markdown(
    """
    <style>
    .gci-header{
      padding:18px;border-radius:12px;
      background: linear-gradient(90deg,#0b0b0b 0%, #1a1a1a 50%, #3a2b10 100%);
      color: #f9f4ef; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .gci-title{font-size:26px;font-weight:800;color:#f5d76e}
    .gci-sub{color:#e6dccd;margin-top:6px}
    </style>
    <div class="gci-header">
      <div class="gci-title">Global Crypto Insight</div>
      <div class="gci-sub">Live market · Top Movers · Fear & Greed · News · Prophet+LSTM Forecast</div>
    </div>
    """, unsafe_allow_html=True
)

st.sidebar.header("Settings")
currency = st.sidebar.selectbox("Display currency", ["USD","CAD","EUR","GBP"], index=0)
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)
st.sidebar.markdown("---")
symbols_default = st.sidebar.text_area("Symbols (comma separated)", value="BTC-USD,ETH-USD,ADA-USD,SOL-USD,XRP-USD").upper()
symbols = [s.strip() for s in symbols_default.split(",") if s.strip()]
st.sidebar.markdown("---")
manual = st.sidebar.text_input("Add single symbol (e.g. ADA-USD):", value="")
if manual:
    mm = manual.strip().upper()
    if mm not in symbols:
        symbols.insert(0, mm)
st.sidebar.markdown("---")
st.sidebar.write("Auto-refresh interval (seconds):", AUTO_REFRESH_SECONDS)
st.sidebar.caption("Prophet and TensorFlow optional. LSTM used only if TF installed and enough history.")

# FX rate helper
@st.cache_data(ttl=300)
def get_fx(target):
    if target == "USD": return 1.0
    mp = {"CAD":"USDCAD=X","EUR":"USDEUR=X","GBP":"USDGBP=X"}
    t = mp.get(target)
    if not t: return 1.0
    try:
        df = yf.download(t, period="5d", interval="1d", progress=False)
        if df is None or df.empty: return 1.0
        return float(df['Close'].dropna().iloc[-1])
    except Exception:
        return 1.0

fx = get_fx(currency)

# Tabs
tabs = st.tabs(["Market","Top Movers","Forecast","Portfolio","News","About"])
tab_market, tab_movers, tab_forecast, tab_portfolio, tab_news, tab_about = tabs

# ---------------- MARKET TAB ----------------
with tab_market:
    st.header("Market Overview")
    fg = fetch_fear_greed()
    c1,c2,c3,c4 = st.columns([1.2,2,2,2])
    with c1:
        if fg['value'] is not None:
            figg = go.Figure(go.Indicator(mode="gauge+number", value=fg['value'],
                                         title={"text":f"Fear & Greed ({fg['date']})"},
                                         gauge={"axis":{"range":[0,100]}, "bar":{"color":"#f5d76e"}}))
            figg.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figg, use_container_width=True)
        else:
            st.info("Fear & Greed: N/A")
    with c2:
        st.markdown("### Snapshot")
        if symbols:
            s0 = symbols[0]
            d0 = fetch_yf(s0, period=period, interval=interval)
            if not d0.empty and 'Close' in d0.columns:
                last = float(d0['Close'].dropna().iloc[-1]) * fx
                prev = float(d0['Close'].dropna().iloc[-2]) * fx if d0['Close'].dropna().shape[0]>=2 else last
                ch = (last - prev)/prev*100 if prev!=0 else 0.0
                st.metric(f"{s0} Latest ({currency})", f"{last:,.2f}", delta=f"{ch:+.2f}%")
            else:
                st.info(f"{s0}: no data")
        else:
            st.info("No symbols configured.")
    with c3:
        st.markdown("### Period High")
        rows=[]
        for s in symbols[:6]:
            d = fetch_yf(s, period=period, interval=interval)
            if not d.empty and 'High' in d.columns:
                rows.append(f"{s}: {float(d['High'].dropna().max())*fx:,.2f} {currency}")
        if rows:
            st.markdown("<br/>".join(rows), unsafe_allow_html=True)
        else:
            st.info("No high data.")
    with c4:
        st.markdown("### Period Low")
        rows=[]
        for s in symbols[:6]:
            d = fetch_yf(s, period=period, interval=interval)
            if not d.empty and 'Low' in d.columns:
                rows.append(f"{s}: {float(d['Low'].dropna().min())*fx:,.2f} {currency}")
        if rows:
            st.markdown("<br/>".join(rows), unsafe_allow_html=True)
        else:
            st.info("No low data.")

    st.markdown("---")
    # summary & signals
    summary=[]
    with st.spinner("Fetching market data..."):
        for s in symbols:
            d = fetch_yf(s, period=period, interval=interval)
            if d.empty or 'Close' not in d.columns:
                summary.append({"Symbol":s,"Price":None,"Change24h":None,"Signal":"NO DATA","Color":"#9e9e9e"})
                continue
            d = d.sort_values('Date').reset_index(drop=True)
            price_usd = float(d['Close'].iloc[-1])
            price = price_usd * fx
            prev = float(d['Close'].iloc[-2]) if d['Close'].dropna().shape[0]>=2 else price_usd
            change24 = (price_usd - prev)/prev*100 if prev!=0 else 0.0
            fc = moving_avg_forecast(d['Close'],1)
            next_fc = float(fc[0]) if len(fc)>0 else None
            signal, color, reason = compute_signal_from_close(d['Close'], next_fc)
            summary.append({"Symbol":s,"Price":price,"Change24h":round(change24,2),"Signal":signal,"Color":color,"Reason":reason})
    df_sum = pd.DataFrame(summary)
    if not df_sum.empty:
        df_sum['PriceStr'] = df_sum['Price'].apply(lambda v: fmt_currency(v,currency) if v is not None else "—")
        df_sum['ChangeStr'] = df_sum['Change24h'].apply(lambda v: f"{v:+.2f}%")
        st.dataframe(df_sum[['Symbol','PriceStr','ChangeStr','Signal']].rename(columns={"Symbol":"Symbol","PriceStr":"Price","ChangeStr":"24h","Signal":"Signal"}), use_container_width=True)
    else:
        st.info("No data.")

# ---------------- TOP MOVERS TAB ----------------
with tab_movers:
    st.header("Top Movers (24h) — Source: CoinGecko")
    movers = fetch_top_movers_coingecko(vs_currency='usd', per_page=COINGECKO_PER_PAGE)
    if movers is None:
        st.error("Failed to fetch movers from CoinGecko.")
    else:
        gainers = movers.get("gainers", [])
        losers = movers.get("losers", [])
        st.subheader("Top Gainers (24h)")
        if gainers:
            rows=[]
            for g in gainers:
                rows.append([g['symbol'], g['name'], f"${g['price']:,.6f}", f"{g['change24']:+.2f}%"])
            st.table(pd.DataFrame(rows, columns=["Symbol","Name","Price (USD)","Change 24h"]))
        else:
            st.info("No gainers found.")
        st.subheader("Top Losers (24h)")
        if losers:
            rows=[]
            for g in losers:
                rows.append([g['symbol'], g['name'], f"${g['price']:,.6f}", f"{g['change24']:+.2f}%"])
            st.table(pd.DataFrame(rows, columns=["Symbol","Name","Price (USD)","Change 24h"]))
    st.markdown("---")
    st.caption("Data refreshes every ~2–3 minutes (CoinGecko caching).")

# ---------------- FORECAST TAB ----------------
with tab_forecast:
    st.header("Forecast (Prophet + optional LSTM)")
    f_sym = st.selectbox("Choose symbol", options=symbols, index=0 if symbols else None)
    f_period = st.selectbox("History period", ["3mo","6mo","1y","2y"], index=0)
    f_interval = st.selectbox("Interval", ["1d","1h"], index=0)
    f_horizon = st.selectbox("Horizon (days)", [3,7,14,30], index=1)

    if f_sym:
        df_f = fetch_yf(f_sym, period=f_period, interval=f_interval)
        if df_f.empty:
            st.warning("No historical data.")
        else:
            st.subheader(f"Historical: {f_sym}")
            st.line_chart(df_f.set_index('Date')['Close'])
            # Decide forecasting method
            forecast_vals = None
            fc_dates = None
            used = "none"
            with st.spinner("Running forecast..."):
                # Try Prophet if available and enough data
                try:
                    if HAS_PROPHET and df_f.shape[0] >= 30:
                        pf = df_f[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
                        m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
                        m.fit(pf)
                        future = m.make_future_dataframe(periods=f_horizon, freq='D')
                        pred = m.predict(future)
                        tail = pred.tail(f_horizon)
                        forecast_vals = tail['yhat'].values
                        fc_dates = tail['ds'].dt.date.values
                        used = "prophet"
                    # If TF available and user prefers LSTM we can run LSTM
                    elif HAS_TF and df_f.shape[0] >= 60:
                        # try LSTM
                        try:
                            lstm_pred = forecast_lstm(df_f['Close'], days=f_horizon, epochs=25, units=32)
                            forecast_vals = lstm_pred
                            last_date = df_f['Date'].iloc[-1]
                            fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]
                            used = "lstm"
                        except Exception as e:
                            # fall back to moving avg
                            forecast_vals = moving_avg_forecast(df_f['Close'], f_horizon)
                            last_date = df_f['Date'].iloc[-1]
                            fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]
                            used = "ma_fallback"
                    else:
                        # fallback moving avg
                        forecast_vals = moving_avg_forecast(df_f['Close'], f_horizon)
                        last_date = df_f['Date'].iloc[-1]
                        fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]
                        used = "ma"
                except Exception as e:
                    # safest fallback
                    forecast_vals = moving_avg_forecast(df_f['Close'], f_horizon)
                    last_date = df_f['Date'].iloc[-1]
                    fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]
                    used = "ma_error"
            st.subheader(f"Forecast method: {used.upper()}")
            fc_table = pd.DataFrame({"Date":fc_dates, "Predicted":np.round(forecast_vals,6)})
            st.table(fc_table)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_f['Date'], y=df_f['Close'], mode='lines', name='History'))
            fig.add_trace(go.Scatter(x=fc_dates, y=forecast_vals, mode='lines+markers', name=f'Forecast {f_horizon}d', line=dict(dash='dash', color='#f5d76e')))
            fig.update_layout(template='plotly_dark', height=520)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- PORTFOLIO TAB ----------------
with tab_portfolio:
    st.header("Portfolio")
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = []
    col1,col2,col3 = st.columns([2,1,1])
    with col1:
        p_sym = st.text_input("Symbol (e.g. ADA-USD):","")
    with col2:
        p_qty = st.number_input("Quantity", min_value=0.0, value=0.0, step=0.01)
    with col3:
        p_cost = st.number_input(f"Cost per unit ({currency})", min_value=0.0, value=0.0, step=0.01)
    if st.button("Add to portfolio"):
        if p_sym.strip()=="" or p_qty<=0 or p_cost<=0:
            st.warning("Provide valid symbol, qty and cost.")
        else:
            st.session_state['portfolio'].append({"symbol":p_sym.strip().upper(),"qty":p_qty,"cost":p_cost,"added":datetime.now().isoformat()})
            st.success("Added.")
    st.subheader("Holdings")
    if len(st.session_state['portfolio'])==0:
        st.info("No holdings.")
    else:
        rows=[]; syms = list({p['symbol'] for p in st.session_state['portfolio']})
        price_map={}
        for s in syms:
            d = fetch_yf(s, period="7d", interval="1d")
            if not d.empty and 'Close' in d.columns:
                price_map[s] = float(d['Close'].dropna().iloc[-1]) * fx
            else:
                price_map[s] = None
        for p in st.session_state['portfolio']:
            cur = price_map.get(p['symbol'], None)
            val = cur * p['qty'] if cur is not None else None
            cost_total = p['cost'] * p['qty']
            pnl = val - cost_total if val is not None else None
            rows.append({"Symbol":p['symbol'],"Qty":p['qty'],"Cost/unit":fmt_currency(p['cost'],currency),"Current":fmt_currency(cur,currency) if cur else "—","Value":fmt_currency(val,currency) if val else "—","P&L":fmt_currency(pnl,currency) if pnl else "—"})
        st.table(pd.DataFrame(rows))
        csv = pd.DataFrame(st.session_state['portfolio']).to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv", mime='text/csv')

# ---------------- NEWS TAB ----------------
with tab_news:
    st.header("News")
    q = st.text_input("Filter news by keyword (optional):", "")
    titles = fetch_news_rss(max_items=NEWS_MAX)
    if titles:
        shown = 0
        for t in titles:
            if q.strip()=="" or q.lower() in t.lower():
                st.write("•", t)
                shown += 1
        if shown==0:
            st.info("No headlines match your filter.")
    else:
        st.info("No news fetched. Try again later or check network.")

# ---------------- ABOUT TAB ----------------
with tab_about:
    st.header("About & Notes")
    st.markdown("""
    **Global Crypto Insight — Upgraded (finalizing)**  
    - RSS News (CoinDesk fallback), Top Movers (CoinGecko), Fear & Greed (alternative.me)  
    - Forecast pipeline: Prophet (preferred) -> LSTM (if TF installed) -> Moving-average fallback  
    - Auto-refresh implemented via cached functions with TTLs (data refresh every few minutes)
    """)
    st.markdown("Run: `streamlit run app.py`")
    st.caption("Educational only — not financial advice. Use at your own risk.")

# Footer + last updated info
st.markdown("---")
st.caption(f"Data caches TTLs: yf={180}s, movers={150}s, news={300}s, fng={600}s. Time now: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
