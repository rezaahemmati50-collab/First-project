# app.py
# AureumPro · Pro Edition (with Fear & Greed + Trend Heatmap)
# English UI, Prophet optional. Run: streamlit run app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="AureumPro · Pro", page_icon="💠", layout="wide")

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

# ---------------- Helpers ----------------
def ensure_flat_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

@st.cache_data(ttl=120)
def fetch_yf(symbol, period="3mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = ensure_flat_columns(df)
        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()

def moving_avg_forecast(series, days):
    if series.dropna().empty:
        return np.array([np.nan]*days)
    last = float(series.dropna().iloc[-1])
    avg_pct = series.pct_change().dropna().mean() if series.pct_change().dropna().shape[0]>0 else 0.0
    return np.array([ last * ((1+avg_pct)**i) for i in range(1, days+1) ])

def fmt_price(x, currency="USD"):
    try:
        return f"{x:,.2f} {currency}"
    except Exception:
        return "—"

# Fetch Fear & Greed index from alternative.me (public)
@st.cache_data(ttl=600)
def fetch_fear_greed():
    url = "https://api.alternative.me/fng/"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            js = r.json()
            if "data" in js and len(js["data"])>0:
                entry = js["data"][0]
                value = int(entry.get("value", 50))
                classification = entry.get("value_classification","Neutral")
                timestamp = int(entry.get("timestamp", 0))
                date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
                return {"value": value, "classification": classification, "date": date}
    except Exception:
        pass
    return {"value": None, "classification": "N/A", "date": None}

# Trend heatmap data: percent daily returns for last N days
def build_trend_heatmap(symbols, lookback_days=14, period="1mo", interval="1d"):
    # We'll fetch enough history to cover lookback_days; using period param that usually covers it
    mat = {}
    dates = None
    for s in symbols:
        df = fetch_yf(s, period=period, interval=interval)
        if df.empty or 'Date' not in df.columns:
            continue
        ser = df.set_index('Date')['Close'].dropna().copy()
        # resample to daily if interval not daily
        ser = ser.resample('D').ffill()
        ser = ser.tail(lookback_days+1)  # need returns, so +1
        if ser.shape[0] < 2:
            continue
        rets = ser.pct_change().dropna()  # length = lookback_days or less
        # align dates
        idx = rets.index
        if dates is None:
            dates = idx
        else:
            dates = dates.intersection(idx)
        mat[s] = rets
    if not mat:
        return None, None, None
    # build DataFrame aligned on shared dates
    df_mat = pd.DataFrame(mat)
    df_mat = df_mat.loc[sorted(df_mat.index)]
    # choose last `lookback_days` columns (dates)
    return df_mat.T, df_mat.index, df_mat

# ---------------- UI Header ----------------
st.markdown(
    """
    <style>
    .header {
        background: linear-gradient(90deg,#0f172a 0%, #0b3c5d 100%);
        padding: 18px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 6px 24px rgba(0,0,0,0.3);
    }
    .title {font-size:26px; font-weight:700}
    .subtitle {color:#cbd5e1; margin-top:6px; font-size:14px}
    </style>
    <div class="header">
      <div class="title">AureumPro · Professional</div>
      <div class="subtitle">Live market, multi-model forecasting, Fear & Greed index and Trend Heatmap</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Controls ----------------
col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.markdown("### Market selection")
    symbols_input = st.text_input("Symbols (comma separated, e.g. BTC-USD,ETH-USD,ADA-USD)", value="BTC-USD,ETH-USD,ADA-USD,SOL-USD,XRP-USD")
with col2:
    period = st.selectbox("History period", ["1mo","3mo","6mo","1y","2y"], index=1)
with col3:
    interval = st.selectbox("Interval", ["1d","1h"], index=0)

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

# KPI row: we'll show F&G, Market summary, and quick forecast horizon selector
fg = fetch_fear_greed()

k1, k2, k3, k4 = st.columns([1.4,2,2,2])
with k1:
    # Fear & Greed gauge
    fg_val = fg.get("value", None)
    if fg_val is not None:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fg_val,
            domain={'x':[0,1],'y':[0,1]},
            title={'text':f"Fear & Greed ({fg.get('date')})", 'font':{'size':14}},
            gauge={'axis':{'range':[0,100]},
                   'bar':{'color':'#f5b041'},
                   'steps':[{'range':[0,25],'color':'#6a3d3d'},
                            {'range':[25,40],'color':'#f39c12'},
                            {'range':[40,60],'color':'#f7dc6f'},
                            {'range':[60,75],'color':'#7fb285'},
                            {'range':[75,100],'color':'#2b9348'}]}
        ))
        fig_g.update_layout(height=200, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_g, use_container_width=True)
    else:
        st.info("Fear & Greed: N/A")

with k2:
    st.markdown("### Quick Market Snapshot")
    # fetch first symbol for snapshot
    if symbols:
        s0 = symbols[0]
        df0 = fetch_yf(s0, period=period, interval=interval)
        if not df0.empty:
            last = float(df0['Close'].dropna().iloc[-1])
            prev = float(df0['Close'].dropna().iloc[-2]) if df0['Close'].dropna().shape[0]>=2 else last
            ch = (last - prev)/prev*100 if prev!=0 else 0
            st.metric(label=f"{s0} Latest price", value=f"${last:,.2f}", delta=f"{ch:+.2f}%")
        else:
            st.info(f"{s0}: no data")
    else:
        st.info("No symbol selected")

with k3:
    st.markdown("### Forecast options")
    horizon = st.selectbox("Forecast horizon (days)", [3,7,14,30], index=1)
    model_choice = st.selectbox("Model", ["Prophet (if available)","MovingAvg fallback"], index=0)

with k4:
    st.markdown("### Visualization")
    heat_days = st.slider("Heatmap lookback (days)", min_value=5, max_value=30, value=14, step=1)

st.markdown("---")

# ---------------- Build summary table ----------------
summary_rows = []
for s in symbols:
    df = fetch_yf(s, period=period, interval=interval)
    if df.empty or 'Close' not in df.columns:
        summary_rows.append({"Symbol": s, "Price": None, "Change24h": None, "Forecast": None})
        continue
    df = df.sort_values('Date').reset_index(drop=True)
    price = float(df['Close'].iloc[-1])
    prev = float(df['Close'].iloc[-2]) if df.shape[0]>=2 else price
    ch24 = (price - prev)/prev*100 if prev!=0 else 0.0

    # forecast quick using model_choice
    fc = None
    if model_choice.startswith("Prophet") and HAS_PROPHET and df.shape[0] >= 3:
        try:
            prophet_df = df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
            m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=horizon, freq='D')
            pred = m.predict(future)
            vals = pred['yhat'].values[-horizon:]
            fc = float(vals[0]) if len(vals)>0 else None
        except Exception:
            fc = None
    if fc is None:
        fc_arr = moving_avg_forecast(df['Close'], horizon)
        fc = float(fc_arr[0]) if len(fc_arr)>0 else None

    summary_rows.append({"Symbol":s, "Price":price, "Change24h":round(ch24,2), "ForecastNext": fc})

summary_df = pd.DataFrame(summary_rows)

# format and display
if not summary_df.empty:
    summary_df['Price_str'] = summary_df['Price'].apply(lambda v: fmt_price(v) if v is not None else "—")
    summary_df['ForecastNext_str'] = summary_df['ForecastNext'].apply(lambda v: f"${v:,.2f}" if v is not None else "—")
    summary_df['Change24h_str'] = summary_df['Change24h'].apply(lambda v: f"{v:+.2f}%")
    st.subheader("Market Summary")
    st.dataframe(summary_df[['Symbol','Price_str','Change24h_str','ForecastNext_str']].rename(columns={
        'Symbol':'Symbol','Price_str':'Price','Change24h_str':'24h','ForecastNext_str':'Next Forecast'
    }), use_container_width=True)
else:
    st.info("No market data to display.")

st.markdown("---")

# ---------------- Trend Heatmap ----------------
st.subheader("Trend Heatmap (daily returns)")
heat_df_t, heat_dates, heat_matrix = build_trend_heatmap(symbols, lookback_days=heat_days, period=period, interval=interval)
if heat_df_t is None:
    st.info("Not enough data to build heatmap. Try longer period or different symbols.")
else:
    # heat_df_t is symbols x dates (returns)
    heat_display = heat_df_t.copy()
    # convert to percent
    heat_pct = heat_display * 100
    # sort symbols by recent performance
    if not heat_pct.empty:
        recent = heat_pct.iloc[:, -1].sort_values(ascending=False)
        heat_pct = heat_pct.loc[recent.index]
    # prepare for plotly heatmap (rows symbols, cols dates)
    z = heat_pct.values
    x = [d.strftime("%Y-%m-%d") for d in heat_pct.columns]
    y = heat_pct.index.tolist()
    fig_h = go.Figure(data=go.Heatmap(
        z=np.round(z,2),
        x=x,
        y=y,
        colorscale='RdYlGn',
        colorbar=dict(title="Daily %")
    ))
    fig_h.update_layout(height=300, xaxis_nticks=len(x))
    st.plotly_chart(fig_h, use_container_width=True)

st.markdown("---")

# ---------------- Detailed single symbol panel ----------------
st.subheader("Symbol Detail & Forecast")
sym_choice = st.selectbox("Choose a symbol", options=symbols, index=0 if len(symbols)>0 else 0)
df_sym = fetch_yf(sym_choice, period=period, interval=interval)
if df_sym.empty:
    st.warning("No data for selected symbol.")
else:
    df_sym = df_sym.sort_values('Date').reset_index(drop=True)
    last_price = float(df_sym['Close'].iloc[-1])
    prev_price = float(df_sym['Close'].iloc[-2]) if df_sym.shape[0]>=2 else last_price
    ch = (last_price - prev_price)/prev_price*100 if prev_price!=0 else 0.0

    # KPIs
    c1,c2,c3 = st.columns(3)
    c1.metric(label=f"{sym_choice} Price", value=f"${last_price:,.2f}", delta=f"{ch:+.2f}%")
    c2.metric(label="Period High", value=f"${df_sym['High'].max():,.2f}")
    c3.metric(label="Period Low", value=f"${df_sym['Low'].min():,.2f}")

    # Indicators
    df_sym['MA20'] = df_sym['Close'].rolling(20, min_periods=1).mean()
    df_sym['MA50'] = df_sym['Close'].rolling(50, min_periods=1).mean()
    if HAS_TA:
        try:
            df_sym['RSI'] = ta.momentum.RSIIndicator(df_sym['Close']).rsi()
            macd = ta.trend.MACD(df_sym['Close'])
            df_sym['MACD_diff'] = macd.macd_diff()
        except Exception:
            df_sym['RSI'] = None
            df_sym['MACD_diff'] = None
    else:
        df_sym['RSI'] = None
        df_sym['MACD_diff'] = None

    # Forecast (full series)
    if model_choice.startswith("Prophet") and HAS_PROPHET and df_sym.shape[0] >= 3:
        try:
            prophet_df = df_sym[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
            m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=horizon, freq='D')
            pred = m.predict(future)
            forecast_vals = pred[['ds','yhat','yhat_lower','yhat_upper']].tail(horizon)
        except Exception:
            forecast_vals = None
    else:
        vals = moving_avg_forecast(df_sym['Close'], horizon)
        last_date = df_sym['Date'].iloc[-1]
        dts = [(last_date + timedelta(days=i+1)).date() for i in range(len(vals))]
        forecast_vals = pd.DataFrame({"ds":dts,"yhat":vals,"yhat_lower":vals,"yhat_upper":vals})

    # Plot
    fig = go.Figure()
    # price line
    fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['Close'], mode='lines', name='Close'))
    # MA lines
    fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['MA20'], mode='lines', name='MA20', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['MA50'], mode='lines', name='MA50', line=dict(color='orange')))
    # forecast line
    try:
        fig.add_trace(go.Scatter(x=forecast_vals['ds'], y=forecast_vals['yhat'], mode='lines+markers', name=f'Forecast {horizon}d', line=dict(dash='dash', color='cyan')))
    except Exception:
        pass
    fig.update_layout(template='plotly_dark', height=520)
    st.plotly_chart(fig, use_container_width=True)

    # show forecast table
    st.subheader("Forecast Values")
    try:
        display_fc = forecast_vals.copy()
        display_fc['yhat'] = display_fc['yhat'].astype(float)
        display_fc['yhat'] = display_fc['yhat'].round(4)
        display_fc = display_fc.rename(columns={'ds':'Date','yhat':'Predicted','yhat_lower':'Low','yhat_upper':'High'})
        st.table(display_fc)
    except Exception:
        st.info("No forecast table available.")

st.markdown("---")
st.caption("This tool is for educational purposes only and not financial advice. Manage your own risk.")
