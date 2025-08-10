# app.py — AureumAI Pro (Streamlit)
# Features:
# - multi-asset support
# - choose forecasting model: Prophet / Moving Average / LSTM (optional)
# - style & color selection for chart lines
# - interactive Plotly charts, download CSVs
# - safe handling of Close column and CSV upload

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io

st.set_page_config(page_title="AureumAI Pro · Crypto Forecast", layout="wide")
st.title("AureumAI Pro · Crypto Forecast")

# -------------------------
# Try optional imports
# -------------------------
HAS_PROPHET = False
HAS_TF = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    HAS_TF = True
except Exception:
    HAS_TF = False

# -------------------------
# Sidebar: controls
# -------------------------
st.sidebar.header("Settings")

ASSETS = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "Solana (SOL)": "SOL-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Dogecoin (DOGE)": "DOGE-USD"
}
asset = st.sidebar.selectbox("Select asset", list(ASSETS.keys()), index=1)
symbol = ASSETS[asset]

period = st.sidebar.selectbox("History period", ["3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("Data interval", ["1d","1wk"], index=0)
forecast_days = st.sidebar.slider("Forecast horizon (days)", 1, 60, 7)

st.sidebar.markdown("### Forecast model")
model_choice = st.sidebar.selectbox(
    "Choose model",
    ("Auto (Prophet → LSTM → MovingAvg)", "Prophet (recommended)", "Moving Average (fast)", "LSTM (optional)")
)
use_log = st.sidebar.checkbox("Use log-transform (stabilize variance)", value=True)

st.sidebar.markdown("### Chart style")
color_actual = st.sidebar.color_picker("Actual line color", "#00BFA6")
color_forecast = st.sidebar.color_picker("Forecast line color", "#FFA500")
line_style = st.sidebar.selectbox("Forecast line style", ["solid","dash","dot"], index=2)

st.sidebar.markdown("---")
st.sidebar.write("Optional packages:")
st.sidebar.write(f"- Prophet: {'✔' if HAS_PROPHET else '✖'}")
st.sidebar.write(f"- TensorFlow (LSTM): {'✔' if HAS_TF else '✖'}")

# CSV upload override
uploaded = st.sidebar.file_uploader("Upload CSV (optional, Date & Close)", type=["csv"])

# -------------------------
# Helpers
# -------------------------
@st.cache_data(ttl=300)
def fetch_data_yf(sym, period, interval):
    try:
        df = yf.download(sym, period=period, interval=interval, progress=False)
        return df
    except Exception:
        return pd.DataFrame()

def normalize_df(df):
    """Ensure Date and Close columns exist and are typed."""
    out = df.copy()
    if np.issubdtype(out.index.dtype, np.datetime64):
        out = out.reset_index()
    # find date col
    date_col = None
    for c in out.columns:
        if str(c).lower() in ("date","datetime","index"):
            date_col = c; break
    if date_col is None:
        date_col = out.columns[0]
    out = out.rename(columns={date_col:"Date"})
    # find close col
    close_col = None
    for c in out.columns:
        if str(c).lower()=="close":
            close_col = c; break
    if close_col is None:
        for c in out.columns:
            if str(c).lower() in ("adj close","adjusted close"):
                close_col = c; break
    if close_col is None:
        numeric = out.select_dtypes(include=[np.number]).columns.tolist()
        if numeric:
            close_col = numeric[-1]
    if close_col is None:
        raise ValueError("Could not find Close column. Upload CSV with Date and Close.")
    out = out.rename(columns={close_col:"Close"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    return out

def moving_avg_forecast(series, days):
    last = series.iloc[-1]
    avg_pct = series.pct_change().mean()
    return [last * ((1+avg_pct)**i) for i in range(1, days+1)]

def prepare_lstm_series(series, n_lags=10, n_epochs=50):
    """Prepare simple LSTM dataset (scaled)"""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    vals = series.values.reshape(-1,1)
    scaled = scaler.fit_transform(vals)
    X, y = [], []
    for i in range(n_lags, len(scaled)):
        X.append(scaled[i-n_lags:i,0])
        y.append(scaled[i,0])
    X = np.array(X); y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

def run_lstm_forecast(series, forecast_days, n_lags=10, epochs=30):
    if not HAS_TF:
        raise RuntimeError("TensorFlow not available.")
    X, y, scaler = prepare_lstm_series(series, n_lags=n_lags, n_epochs=epochs)
    if X.shape[0] < 10:
        raise RuntimeError("Not enough data for LSTM.")
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1],1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=epochs, batch_size=16, verbose=0)
    # iterative forecasting
    last_window = X[-1,:,0].tolist()
    preds = []
    for i in range(forecast_days):
        inp = np.array(last_window[-n_lags:]).reshape((1,n_lags,1))
        p = model.predict(inp, verbose=0)[0,0]
        preds.append(p)
        last_window.append(p)
    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    return preds

def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# -------------------------
# Load data
# -------------------------
if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        st.sidebar.success("Using uploaded CSV")
    except Exception as e:
        st.sidebar.error(f"CSV read error: {e}")
        raw = pd.DataFrame()
else:
    raw = fetch_data_yf(symbol, period, interval)

if raw is None or raw.empty:
    st.error("No data available. Try different period or upload CSV.")
    st.stop()

try:
    df = normalize_df(raw)
except Exception as e:
    st.error(f"Data normalization error: {e}")
    st.stop()

# -------------------------
# Show summary
# -------------------------
st.subheader(f"{asset} · {symbol}")
st.write(f"History: {df['Date'].min().date()} → {df['Date'].max().date()}  •  {len(df)} rows")
st.dataframe(df.tail(5))

# -------------------------
# Choose model (resolve Auto)
# -------------------------
chosen = model_choice
if model_choice.startswith("Auto"):
    if HAS_PROPHET:
        chosen = "Prophet (recommended)"
    elif HAS_TF:
        chosen = "LSTM (optional)"
    else:
        chosen = "Moving Average (fast)"
st.info(f"Selected model: {chosen}")

# -------------------------
# Compute forecasts
# -------------------------
future_dates = [df['Date'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days+1)]
forecast_vals = None
forecast_df = None

try:
    if chosen.startswith("Prophet") and HAS_PROPHET:
        # prepare prophet df
        p_df = df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'}).copy()
        if use_log and (p_df['y']>0).all():
            p_df['y'] = np.log(p_df['y'])
            log_used = True
        else:
            log_used = False
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, changepoint_prior_scale=0.05)
        m.fit(p_df.rename(columns={'y':'y'}))
        future = m.make_future_dataframe(periods=forecast_days, freq='D')
        pred = m.predict(future)
        if log_used:
            pred['yhat_final'] = np.exp(pred['yhat'])
            pred['yhat_lower_final'] = np.exp(pred['yhat_lower'])
            pred['yhat_upper_final'] = np.exp(pred['yhat_upper'])
        else:
            pred['yhat_final'] = pred['yhat']
            pred['yhat_lower_final'] = pred['yhat_lower']
            pred['yhat_upper_final'] = pred['yhat_upper']
        forecast_vals = pred['yhat_final'].tail(forecast_days).values
        forecast_df = pred[['ds','yhat_final','yhat_lower_final','yhat_upper_final']].tail(forecast_days).rename(
            columns={'ds':'Date','yhat_final':'yhat','yhat_lower_final':'yhat_lower','yhat_upper_final':'yhat_upper'}
        )
    elif chosen.startswith("LSTM") and HAS_TF:
        preds = run_lstm_forecast(df['Close'], forecast_days, n_lags=10, epochs=25)
        forecast_vals = np.array(preds)
        forecast_df = pd.DataFrame({'Date':[d.date() for d in future_dates], 'yhat':forecast_vals})
    else:
        # Moving average fallback / fast model
        forecast_vals = np.array(moving_avg_forecast(df['Close'], forecast_days))
        forecast_df = pd.DataFrame({'Date':[d.date() for d in future_dates], 'yhat':forecast_vals})
except Exception as e:
    st.error(f"Forecasting error: {e}")
    st.info("Falling back to Moving Average forecast.")
    forecast_vals = np.array(moving_avg_forecast(df['Close'], forecast_days))
    forecast_df = pd.DataFrame({'Date':[d.date() for d in future_dates], 'yhat':forecast_vals})

# -------------------------
# Plot interactive chart
# -------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual', line=dict(color=color_actual, width=2)))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_vals, mode='lines+markers', name='Forecast',
                         line=dict(color=color_forecast, width=2, dash=line_style)))

# optionally show uncertainty if available
if forecast_df is not None and 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
    fig.add_trace(go.Scatter(
        x=list(pd.to_datetime(forecast_df['Date'])) + list(pd.to_datetime(forecast_df['Date'])[::-1]),
        y=list(forecast_df['yhat_upper']) + list(forecast_df['yhat_lower'][::-1]),
        fill='toself',
        fillcolor='rgba(200,200,200,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))

fig.update_layout(template="plotly_dark", title=f"{asset} — {symbol} — Forecast ({forecast_days} days)", height=600,
                  xaxis_title="Date", yaxis_title="Price (quote currency)")

st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Show forecast table and downloads
# -------------------------
st.markdown("### Forecast (next days)")
if forecast_df is None:
    forecast_df = pd.DataFrame({'Date':[d.date() for d in future_dates], 'yhat':forecast_vals})
st.dataframe(forecast_df.reset_index(drop=True), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download raw data CSV", to_csv_bytes(df), file_name=f"{symbol}_raw.csv")
with col2:
    st.download_button("Download forecast CSV", to_csv_bytes(forecast_df), file_name=f"{symbol}_forecast.csv")

st.markdown("---")
st.caption("AureumAI Pro — Demo. Not financial advice. For production, secure API keys, rate limits and model validation are required.")
