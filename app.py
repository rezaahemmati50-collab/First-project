# app.py
"""
AureumAI — Professional multi-asset crypto forecasting demo (Streamlit)
Features:
- Multi-select assets (BTC, ETH, ...)
- Date range selection
- USD/CAD choice via symbol suffix
- Prophet forecasting with log-transform & changepoint slider
- Forecast horizon selectable (1-60 days)
- Auto-detect Close/Adj Close/close column names
- Dark/Light theme (simple CSS)
- Download raw & forecast CSVs per asset
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import datetime, timedelta
import io

# ---------------------------
# Page config & basic style
# ---------------------------
st.set_page_config(page_title="AureumAI · Crypto Forecast", layout="wide", initial_sidebar_state="expanded")

# basic colors
PRIMARY_BG = "#0E1117"
TEXT_COLOR = "#E8EEF1"
ACCENT = "#D4AF37"  # gold

# sidebar controls (theme, assets, dates, forecast settings)
st.sidebar.markdown("## ⚙️ Settings")

theme = st.sidebar.selectbox("Theme", ["Dark (recommended)", "Light"])

if theme.startswith("Dark"):
    st.markdown(
        f"""
        <style>
        .stApp {{ background-color: {PRIMARY_BG}; color: {TEXT_COLOR}; }}
        .block-container {{ padding: 1rem 2rem; }}
        h1 {{ color: {ACCENT}; }}
        .stButton>button {{ background-color: {ACCENT}; color: #000; border-radius:6px; }}
        .stMarkdown p {{ color: {TEXT_COLOR}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stApp {{ background-color: #FFFFFF; color: #0b0b0b; }}
        h1 {{ color: #0b7a75; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

st.markdown(f"<h1 style='text-align:center;'>AureumAI · Crypto Forecast</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color: #bdbdbd;'>Accurate, elegant and fast crypto forecasts — demo</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------------------
# Asset list & selection
# ---------------------------
STOCKS = {
    "Bitcoin (BTC)": "BTC",
    "Ethereum (ETH)": "ETH",
    "Cardano (ADA)": "ADA",
    "Stellar (XLM)": "XLM",
    "Solana (SOL)": "SOL",
    "Litecoin (LTC)": "LTC",
    "Ripple (XRP)": "XRP",
    "Dogecoin (DOGE)": "DOGE",
    "Polkadot (DOT)": "DOT"
}

st.sidebar.markdown("### Select assets")
selected_names = st.sidebar.multiselect("Choose assets (multi-select supported):", list(STOCKS.keys()), default=["Ethereum (ETH)"])

# currency suffix
currency = st.sidebar.selectbox("Currency / quote (price in):", ["USD", "CAD"], index=0)

# date range
st.sidebar.markdown("### Date range (history for training)")
default_end = datetime.utcnow().date()
default_start = default_end - timedelta(days=365)  # 1 year by default
start_date = st.sidebar.date_input("Start date", default_start)
end_date = st.sidebar.date_input("End date", default_end)
if start_date >= end_date:
    st.sidebar.error("Start date must be before End date.")

# forecast controls
st.sidebar.markdown("### Forecast settings")
forecast_days = st.sidebar.slider("Forecast horizon (days):", 1, 60, 7)
changepoint_prior_scale = st.sidebar.slider("Model sensitivity (changepoint_prior_scale)", 0.001, 0.5, 0.05, step=0.01)
use_log_transform = st.sidebar.checkbox("Use log-transform (stabilize variance) — recommended", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Powered by Prophet • Demo")

# ---------------------------
# Utility functions
# ---------------------------

@st.cache_data(ttl=300)
def fetch_yahoo(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """
    Fetch data from yfinance. Returns DataFrame (index may be DatetimeIndex).
    """
    try:
        # yfinance accepts start/end as strings or dates
        raw = yf.download(symbol, start=start, end=end + timedelta(days=1), progress=False)
        if raw is None:
            return pd.DataFrame()
        return raw
    except Exception as e:
        st.error(f"Error fetching {symbol}: {e}")
        return pd.DataFrame()

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize columns:
    - ensure Date column exists as 'Date' (from index or common names)
    - find 'Close' column case-insensitively or fall back to 'Adj Close' or last numeric column
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    # If index is datetime, reset to column
    if np.issubdtype(out.index.dtype, np.datetime64):
        out = out.reset_index()

    # find date column
    date_col = None
    for c in out.columns:
        if str(c).lower() in ("date", "datetime", "index"):
            date_col = c
            break
    if date_col is None:
        # fallback: first column if it's datetime-like
        if np.issubdtype(out.columns[0].dtype, np.datetime64):
            date_col = out.columns[0]
        else:
            # try to coerce first column to datetime
            try:
                out[out.columns[0]] = pd.to_datetime(out[out.columns[0]])
                date_col = out.columns[0]
            except Exception:
                date_col = out.columns[0]

    out = out.rename(columns={date_col: "Date"})

    # find close column
    close_col = None
    for c in out.columns:
        if str(c).lower() == "close":
            close_col = c
            break
    if close_col is None:
        for c in out.columns:
            if str(c).lower() in ("adj close", "adjusted close"):
                close_col = c
                break
    if close_col is None:
        # fallback: last numeric column
        numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            close_col = numeric_cols[-1]

    if close_col is None:
        raise ValueError("Could not find a 'Close' column in fetched data. Provide a CSV with Date & Close or try another symbol.")

    out = out.rename(columns={close_col: "Close"})
    # coerce types
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return out

def run_prophet_and_forecast(df: pd.DataFrame, days: int, changepoint_prior_scale: float, use_log: bool):
    """
    Fit Prophet and forecast. Returns (model, forecast_df, train_df)
    - train_df has columns ds, y (original scale)
    - forecast_df contains yhat_final, yhat_lower_final, yhat_upper_final
    """
    train_df = df[["Date", "Close"]].rename(columns={"Date":"ds", "Close":"y"}).copy()
    train_df = train_df.sort_values("ds").reset_index(drop=True)

    # optional log transform
    if use_log:
        train_df["y_trans"] = np.log(train_df["y"].replace(0, np.nan)).dropna()
        # ensure no -inf/inf
        train_fit = train_df.dropna(subset=["y_trans"])[["ds","y_trans"]].rename(columns={"y_trans":"y"})
    else:
        train_fit = train_df[["ds","y"]].rename(columns={"y":"y"})

    # minimal rows check
    if train_fit.shape[0] < 5:
        raise ValueError("Not enough history to fit the model (need at least ~5 rows).")

    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False,
                changepoint_prior_scale=changepoint_prior_scale, seasonality_mode='multiplicative')
    m.fit(train_fit)

    future = m.make_future_dataframe(periods=days, freq='D')
    forecast = m.predict(future)

    # revert log if used
    if use_log:
        forecast["yhat_final"] = np.exp(forecast["yhat"])
        forecast["yhat_lower_final"] = np.exp(forecast["yhat_lower"])
        forecast["yhat_upper_final"] = np.exp(forecast["yhat_upper"])
    else:
        forecast["yhat_final"] = forecast["yhat"]
        forecast["yhat_lower_final"] = forecast["yhat_lower"]
        forecast["yhat_upper_final"] = forecast["yhat_upper"]

    return m, forecast, train_df

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------------------------
# Main layout: left column controls, right column results
# ---------------------------
left_col, right_col = st.columns([1, 3])

with left_col:
    st.header("Workspace")
    st.write("Selected assets:")
    for name in selected_names:
        st.write(f"- **{name}**")
    st.markdown("---")
    st.write("History window:")
    st.write(f"- From: **{start_date}**  →  To: **{end_date}**")
    st.write(f"Forecast horizon: **{forecast_days}** days")
    st.write(f"Currency: **{currency}**")
    st.write(f"Model sensitivity: **{changepoint_prior_scale}**")
    st.markdown("---")
    if st.button("🔁 Refresh & Run Forecast"):
        st.experimental_rerun()

with right_col:
    if not selected_names:
        st.warning("Please select at least one asset from the sidebar.")
        st.stop()

    # We'll iterate assets and render a card per asset
    for name in selected_names:
        symbol_root = STOCKS.get(name, None)
        if symbol_root is None:
            st.error(f"Unknown asset: {name}")
            continue
        full_symbol = f"{symbol_root}-{currency}"

        st.markdown(f"## {name}  ·  `{full_symbol}`")
        with st.spinner(f"Fetching {full_symbol} ..."):
            raw = fetch_yahoo(full_symbol, start_date, end_date)

        if raw is None or raw.empty:
            st.error(f"No data fetched for {full_symbol}. Try different dates or check internet.")
            continue

        try:
            df_norm = ensure_columns(raw)
        except Exception as e:
            st.error(f"Data normalization error for {full_symbol}: {e}")
            continue

        # show last rows and small stats
        st.markdown("**Recent data**")
        st.dataframe(df_norm[["Date","Close"]].tail(8), use_container_width=True)
        st.metric("Latest Close", f"${df_norm['Close'].iloc[-1]:,.2f}")

        # historical chart (plotly)
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Scatter(x=df_norm["Date"], y=df_norm["Close"], mode="lines", name="Close",
                                     line=dict(color=ACCENT)))
        hist_fig.update_layout(template="plotly_dark" if theme.startswith("Dark") else "plotly_white",
                               margin=dict(l=20,r=20,t=30,b=20),
                               height=260,
                               xaxis_title="", yaxis_title="Price")
        st.plotly_chart(hist_fig, use_container_width=True)

        # run prophet
        with st.spinner("Fitting model and forecasting..."):
            try:
                model, forecast_df, train_df = run_prophet_and_forecast(df_norm, forecast_days, changepoint_prior_scale, use_log_transform)
            except Exception as e:
                st.error(f"Model error for {full_symbol}: {e}")
                continue

        # prepare small result table (only future days)
        predicted = forecast_df[["ds","yhat_final","yhat_lower_final","yhat_upper_final"]].tail(forecast_days).copy()
        predicted = predicted.rename(columns={"ds":"Date","yhat_final":"yhat","yhat_lower_final":"yhat_lower","yhat_upper_final":"yhat_upper"})
        predicted["Date"] = pd.to_datetime(predicted["Date"]).dt.date
        predicted[["yhat","yhat_lower","yhat_upper"]] = predicted[["yhat","yhat_lower","yhat_upper"]].round(2)

        st.markdown("### Forecast (table)")
        st.dataframe(predicted, use_container_width=True)

        # metrics: last close vs next-day forecast
        last_close = train_df["y"].iloc[-1]
        next_day_forecast = predicted["yhat"].iloc[0] if not predicted.empty else None
        col_a, col_b = st.columns(2)
        col_a.metric("Last Close", f"${last_close:,.2f}")
        if next_day_forecast is not None:
            col_b.metric(f"Forecast (next day)", f"${next_day_forecast:,.2f}", delta=f"${(next_day_forecast - last_close):.2f}")

        # interactive forecast plot using Prophet's forecast but with our final columns
        st.markdown("### Forecast chart (with uncertainty)")
        fig = go.Figure()
        # historical
        fig.add_trace(go.Scatter(x=train_df["ds"], y=train_df["y"], mode="lines", name="Actual", line=dict(color="#888888")))
        # forecast line
        fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_final"], mode="lines", name="Forecast", line=dict(color=ACCENT, width=2)))
        # uncertainty band
        fig.add_trace(go.Scatter(
            x=list(forecast_df["ds"]) + list(forecast_df["ds"][::-1]),
            y=list(forecast_df["yhat_upper_final"]) + list(forecast_df["yhat_lower_final"][::-1]),
            fill='toself',
            fillcolor='rgba(150,150,150,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            name="Uncertainty"
        ))
        fig.update_layout(template="plotly_dark" if theme.startswith("Dark") else "plotly_white",
                          margin=dict(l=10,r=10,t=30,b=10), height=420,
                          xaxis_title="", yaxis_title="Price (quote currency)")
        st.plotly_chart(fig, use_container_width=True)

        # downloads
        st.markdown("**Downloads**")
        raw_csv = to_csv_bytes(df_norm)
        forecast_csv = to_csv_bytes(forecast_df[["ds","yhat_final","yhat_lower_final","yhat_upper_final"]])
        st.download_button(f"Download raw data ({full_symbol}) CSV", raw_csv, file_name=f"{full_symbol}_raw.csv", mime="text/csv")
        st.download_button(f"Download forecast ({full_symbol}) CSV", forecast_csv, file_name=f"{full_symbol}_forecast.csv", mime="text/csv")

        st.markdown("---")

# ---------------------------
# footer
# ---------------------------
st.markdown("**Note:** Forecasts are generated using historical data and Prophet. They are probabilistic and not financial advice.")
st.caption("AureumAI · Demo — For production use: consider adding ensemble models (ARIMA/LSTM), authentication, rate limits & licensing.")
