# app.py
"""
Crypto Market Analyzer — Final Integrated Single-file App
Features:
- Attractive header banner (place images/header.png if you have it)
- Multi-select cryptos + manual input
- Choose display currency: USD / CAD / GBP / EUR (best-effort conversion)
- Fear & Greed index (sample lookup)
- News (local sample fallback; optional NewsAPI key)
- Technical indicators: MA20/50/200, RSI14, MACD diff
- Ensemble Buy/Hold/Sell signal
- Prophet forecasting (optional; auto-disabled if prophet not installed)
- CSV downloads and defensive error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import os
import requests
import io

# Try Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# --------- Page config & header style ----------
st.set_page_config(page_title="Crypto Market Analyzer", page_icon="🚀", layout="wide")
st.markdown("""
    <style>
    .header-box{display:flex;gap:18px;padding:12px;border-radius:12px;background:linear-gradient(90deg,#0b1220,#07112a);color:#fff}
    .logo-box{width:96px;height:96px;border-radius:12px;background:linear-gradient(135deg,#f6d365,#fda085);display:flex;align-items:center;justify-content:center;font-weight:800;color:#07112a;font-size:28px}
    .title{font-size:26px;margin:0;font-weight:800}
    .subtitle{font-size:13px;color:#d1d5db;margin-top:6px}
    .small{font-size:12px;color:#9ca3af}
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 7])
with col1:
    # try to load header image if exists, else show logo box
    if os.path.exists("images/header.png"):
        st.image("images/header.png", width=110)
    else:
        st.markdown('<div class="logo-box">CMA</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="header-box"><div><h1 class="title">Crypto Market Analyzer</h1>'
                '<div class="subtitle">Live prices · Technical indicators · Signals · Forecasting (optional)</div>'
                '<div class="small">Not financial advice. Use responsibly.</div></div></div>', unsafe_allow_html=True)

st.write("---")

# --------- Helpers & utilities ----------
@st.cache_data(ttl=120)
def yf_download(ticker, period="3mo", interval="1d"):
    """Download via yfinance and return df (or empty df + error)"""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None:
            return pd.DataFrame(), "yfinance returned None"
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"yfinance error: {e}"

def extract_close(df, prefer_ticker=None):
    """Robustly extract Close series (1d) from yfinance DataFrame (handles MultiIndex)."""
    if df is None or df.empty:
        return None, "empty dataframe"
    # MultiIndex handling
    if isinstance(df.columns, pd.MultiIndex):
        try:
            lvl0 = list(df.columns.get_level_values(0))
            lvl1 = list(df.columns.get_level_values(1))
            # try level name 'Close' in level 0
            if 'Close' in lvl0:
                cands = [c for c in df.columns if c[0] == 'Close']
                if prefer_ticker:
                    for c in cands:
                        if prefer_ticker in str(c[1]):
                            s = df[c].copy(); s.index = pd.to_datetime(s.index); s.name='Close'; return s.dropna(), ""
                s = df[cands[0]].copy(); s.index = pd.to_datetime(s.index); s.name='Close'; return s.dropna(), ""
            # try 'Close' in level 1
            if 'Close' in lvl1:
                cands = [c for c in df.columns if c[1] == 'Close']
                s = df[cands[0]].copy(); s.index = pd.to_datetime(s.index); s.name='Close'; return s.dropna(), ""
        except Exception as e:
            return None, f"multiindex extraction error: {e}"
        return None, "MultiIndex but Close not found"
    # normal columns
    for cand in ['Close','close','Adj Close','Adj_Close','AdjClose']:
        if cand in df.columns:
            s = df[cand].copy(); s.index = pd.to_datetime(s.index); s.name='Close'
            return s.dropna(), ""
    return None, "No Close column"

def to_numeric_safe(x):
    try:
        return pd.to_numeric(x, errors='coerce')
    except Exception:
        return pd.Series(dtype='float64')

def compute_indicators(series):
    """Return df with Close, MA20, MA50, MA200, RSI14, MACD_diff."""
    s = to_numeric_safe(series).ffill().dropna()
    if s.empty:
        return pd.DataFrame()
    df = pd.DataFrame({'Close': s})
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(200, min_periods=1).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0); down = -delta.clip(upper=0)
    ema_up = up.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    ema_down = down.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = ema_up / ema_down.replace(0, np.nan)
    df['RSI14'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = macd - signal
    return df

def ensemble_signal(df_ind):
    """Simple ensemble rule returning STRONG BUY/BUY/HOLD/SELL/STRONG SELL."""
    if df_ind.empty:
        return "NO DATA"
    last = df_ind.iloc[-1]
    score = 0
    try:
        if last['Close'] > last['MA20'] > last['MA50']:
            score += 1
        elif last['Close'] < last['MA20'] < last['MA50']:
            score -= 1
    except Exception:
        pass
    rsi = last.get('RSI14', np.nan)
    if not np.isnan(rsi):
        if rsi < 30: score += 1
        elif rsi > 70: score -= 1
    macd = last.get('MACD_diff', np.nan)
    if not np.isnan(macd):
        score += 1 if macd > 0 else -1
    if score >= 2: return "STRONG BUY"
    if score == 1: return "BUY"
    if score == 0: return "HOLD"
    if score == -1: return "SELL"
    return "STRONG SELL"

def try_prophet_forecast(series, days=3):
    """Run Prophet forecast if available, otherwise raise RuntimeError."""
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet not installed in environment.")
    df = series.dropna().to_frame('y').reset_index().rename(columns={series.index.name or series.name:'ds','y':'y'})
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['ds','y'])
    if df.shape[0] < 2:
        raise ValueError("Not enough data for Prophet (need >=2 rows).")
    m = Prophet(daily_seasonality=True)
    m.fit(df[['ds','y']])
    future = m.make_future_dataframe(periods=days)
    fc = m.predict(future)
    return fc[['ds','yhat','yhat_lower','yhat_upper']].tail(days)

# News sample & Fear&Greed sample (static fallback)
SAMPLE_NEWS = [
    {"date":"2025-08-01","title":"Market update: BTC rises","source":"CryptoNews","url":"https://example.com/1"},
    {"date":"2025-07-30","title":"Ethereum upgrade announced","source":"CoinDaily","url":"https://example.com/2"},
]
# small sample F&G (value 0-100)
SAMPLE_FG = {"value": 55, "label": "Neutral"}  # you can replace by live source if desired

# --------- Sidebar inputs ----------
st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["Manual tickers", "Upload CSV (ds,y)"])
if mode == "Manual tickers":
    default = "BTC-USD,ETH-USD,ADA-USD"
    user_input = st.sidebar.text_input("Tickers (comma separated)", value=default)
    tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]
else:
    uploaded = st.sidebar.file_uploader("Upload CSV (ds,y)", type=["csv"])

history = st.sidebar.selectbox("History", ["1mo","3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)
display_currency = st.sidebar.selectbox("Display currency", ["USD","CAD","GBP","EUR"], index=0)
forecast_days = st.sidebar.selectbox("Forecast days", [3,7,30], index=0)
enable_prophet = st.sidebar.checkbox("Enable Prophet forecasting (optional)", value=PROPHET_AVAILABLE)
news_api_key = st.sidebar.text_input("NewsAPI key (optional)", value="")
show_news = st.sidebar.checkbox("Show news & Fear&Greed", value=True)
download_csv = st.sidebar.checkbox("Enable CSV downloads", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: If Prophet not installed, disable forecasting or install 'prophet' package.")

# --------- Main layout: top row with FG index and news ----------
top_cols = st.columns([2, 4, 6])
with top_cols[0]:
    st.subheader("Fear & Greed")
    if show_news:
        try:
            # placeholder: display sample. Could be replaced by live API if configured.
            st.metric(label="Fear & Greed Index", value=f"{SAMPLE_FG['value']}", delta=SAMPLE_FG.get('label',''))
        except Exception:
            st.write("N/A")
with top_cols[1]:
    st.subheader("Quick Controls")
    st.write(f"Display currency: **{display_currency}**")
    st.write(f"History: **{history}** | Interval: **{interval}**")
with top_cols[2]:
    st.subheader("Latest News")
    if show_news:
        # try NewsAPI if key, else sample
        if news_api_key:
            try:
                url = "https://newsapi.org/v2/everything"
                params = {"q":"crypto OR bitcoin OR ethereum","pageSize":5,"language":"en","sortBy":"publishedAt","apiKey":news_api_key}
                r = requests.get(url, params=params, timeout=8)
                if r.status_code == 200:
                    arts = r.json().get("articles",[])
                    small = [{"date":a.get("publishedAt"), "title":a.get("title"), "source":a.get("source",{}).get("name"), "url":a.get("url")} for a in arts]
                    st.table(pd.DataFrame(small))
                else:
                    st.table(pd.DataFrame(SAMPLE_NEWS))
            except Exception:
                st.table(pd.DataFrame(SAMPLE_NEWS))
        else:
            st.table(pd.DataFrame(SAMPLE_NEWS))

st.write("---")

# --------- Main analysis ----------
if mode == "Upload CSV (ds,y)":
    if uploaded is None:
        st.info("Upload CSV with columns ds (date) and y (price).")
        st.stop()
    try:
        df_csv = pd.read_csv(uploaded)
        cols_lower = {c.lower(): c for c in df_csv.columns}
        if 'ds' in cols_lower and 'y' in cols_lower:
            ds_col = cols_lower['ds']; y_col = cols_lower['y']
            df_csv[ds_col] = pd.to_datetime(df_csv[ds_col], errors='coerce')
            df_csv[y_col] = pd.to_numeric(df_csv[y_col], errors='coerce')
            df_csv = df_csv.dropna(subset=[ds_col,y_col]).set_index(ds_col).sort_index()
            series = df_csv[y_col]
            df_ind = compute_indicators(series)
            st.subheader("Uploaded data indicators")
            st.dataframe(df_ind.tail(10))
            st.markdown(f"**Signal:** {ensemble_signal(df_ind)}")
            if enable_prophet and PROPHET_AVAILABLE:
                try:
                    fc = try_prophet_forecast(series, days=forecast_days)
                    fc['ds'] = pd.to_datetime(fc['ds']).dt.date
                    st.subheader("Forecast")
                    st.table(fc)
                    if download_csv:
                        st.download_button("Download forecast CSV", fc.to_csv(index=False).encode('utf-8'), file_name="uploaded_forecast.csv", mime='text/csv')
                except Exception as e:
                    st.warning(f"Forecast error: {e}")
        else:
            st.error("CSV must contain ds and y columns.")
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")

else:
    st.subheader("Select tickers")
    sel = st.multiselect("Tickers", options=tickers, default=tickers[:2])
    if not sel:
        st.info("Select at least one ticker.")
        st.stop()

    all_combined = []
    for t in sel:
        st.markdown("---")
        st.markdown(f"### {t} — display in {display_currency}")
        df_raw, err = yf_download(t, period=history, interval=interval)
        if err:
            st.error(f"Failed to fetch {t}: {err}")
            continue
        # Try direct paired ticker in chosen fiat (e.g., BTC-CAD). If exists, use it.
        base = t.split("-")[0]
        direct_pair = f"{base}-{display_currency}"
        close_ser = None
        dbg_msg = ""
        df_direct, err_direct = yf_download(direct_pair, period=history, interval=interval)
        if err_direct is None and (not df_direct.empty):
            close_ser, dbg_msg = extract_close(df_direct, prefer_ticker=direct_pair)
            if close_ser is None:
                dbg_msg = f"Direct pair {direct_pair} returned, but extraction failed: {dbg_msg}"
        # fallback: use USD pair and try get FX conversion if needed
        if close_ser is None:
            close_ser_usd, dbg_usd = extract_close(df_raw, prefer_ticker=t)
            if close_ser_usd is None:
                st.error(f"Could not extract Close for {t}. Debug: {dbg_usd}")
                continue
            # if display currency is USD, good
            if display_currency == "USD":
                close_ser = close_ser_usd
            else:
                # try to get FX rate via yfinance for pair like "USD{CUR}=X" or "{CUR}=X"
                fx_tickers = [f"US{display_currency}=X", f"{display_currency}=X", f"USD{display_currency}=X", f"{display_currency}USD=X"]
                fx_rate = None
                for fx in fx_tickers:
                    df_fx, err_fx = yf_download(fx, period="7d", interval="1d")
                    if err_fx is None and not df_fx.empty:
                        sfx, _ = extract_close(df_fx)
                        if sfx is not None and not sfx.empty:
                            # take last close
                            try:
                                val = float(sfx.iloc[-1])
                                fx_rate = val
                                dbg_msg = f"Using FX {fx} rate {val}"
                                break
                            except Exception:
                                continue
                if fx_rate is None:
                    st.warning(f"Could not find FX pair for converting to {display_currency}. Showing USD prices.")
                    close_ser = close_ser_usd
                else:
                    # interpret fx_rate: many tickers have format like USD/CAD or CAD/USD depending on ticker.
                    # We can't guarantee orientation, so attempt reasonable conversion:
                    # If fx_rate > 5 or fx_rate < 0.2 unusual — still use as multiplier.
                    # Convert USD price -> target currency by multiplying by fx_rate if fx ticker is USD to CUR,
                    # otherwise divide. This is heuristic; we inform user.
                    # We'll attempt to detect orientation: if fx ticker contains 'USD' first we assume 1 USD = fx_rate CUR.
                    orientation = "unknown"
                    if fx.upper().startswith("USD") or fx.upper().startswith("US"):
                        orientation = "USD_TO_CUR"
                    elif fx.upper().endswith("USD=X") or fx.upper().endswith("USD"):
                        orientation = "CUR_TO_USD"
                    # apply conversion
                    if orientation == "USD_TO_CUR":
                        close_ser = close_ser_usd * fx_rate
                    elif orientation == "CUR_TO_USD":
                        # close_ser_usd is USD price; to get CUR = USD / (CUR per USD) -> divide
                        try:
                            close_ser = close_ser_usd / fx_rate
                        except Exception:
                            close_ser = close_ser_usd
                    else:
                        close_ser = close_ser_usd * fx_rate
                    st.info(f"Conversion applied ({dbg_msg}). If numbers look wrong, try display currency USD.")
        # ensure we have series now
        if close_ser is None or close_ser.dropna().empty:
            st.error(f"No close series available for {t}. {dbg_msg}")
            continue

        # compute indicators
        df_ind = compute_indicators(close_ser)
        if df_ind.empty:
            st.error("Not enough data for indicators.")
            continue

        # metrics & signal
        last = df_ind.iloc[-1]
        try:
            last_price = float(last['Close'])
            st.metric(label=f"{t} Last Close ({display_currency})", value=f"{last_price:,.6f}")
        except Exception:
            st.metric(label=f"{t} Last Close ({display_currency})", value="N/A")
        signal = ensemble_signal(df_ind)
        st.markdown(f"**Signal:** `{signal}`")

        # small table and plots
        st.dataframe(df_ind.tail(8).reset_index().rename(columns={df_ind.index.name or 'index':'Date'}))
        # forecast
        forecast_df = None
        if enable_prophet and PROPHET_AVAILABLE:
            try:
                forecast_df = try_prophet_forecast(close_ser, days=forecast_days)
            except Exception as e:
                st.warning(f"Prophet forecast error: {e}")
                forecast_df = None

        # plot main
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['Close'], name='Close'))
            if 'MA20' in df_ind.columns: fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MA20'], name='MA20'))
            if 'MA50' in df_ind.columns: fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MA50'], name='MA50'))
            if forecast_df is not None and not forecast_df.empty:
                fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast', line=dict(dash='dash')))
            fig.update_layout(title=f"{t} price ({display_currency})", xaxis_rangeslider_visible=True, height=460)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plot error: {e}")

        # RSI and MACD mini-charts
        try:
            fig_rsi = go.Figure(); fig_rsi.add_trace(go.Scatter(x=df_ind.index, y=df_ind['RSI14'], name='RSI14'))
            fig_rsi.update_layout(title='RSI (14)', yaxis=dict(range=[0,100]), height=240)
            st.plotly_chart(fig_rsi, use_container_width=True)
        except Exception:
            pass
        try:
            fig_macd = go.Figure(); fig_macd.add_trace(go.Bar(x=df_ind.index, y=df_ind['MACD_diff'], name='MACD diff'))
            fig_macd.update_layout(title='MACD diff', height=240)
            st.plotly_chart(fig_macd, use_container_width=True)
        except Exception:
            pass

        # downloads
        if download_csv:
            try:
                csv_bytes = df_ind.to_csv(index=True).encode('utf-8')
                st.download_button(f"Download {t} indicators CSV", csv_bytes, file_name=f"{t}_indicators.csv", mime='text/csv')
            except Exception:
                pass

        # collect for combined download
        tail = df_ind.tail(5).reset_index().rename(columns={df_ind.index.name or 'index':'Date'})
        tail['ticker'] = t
        all_combined.append(tail[['ticker','Date','Close','MA20','MA50','RSI14','MACD_diff']])

    # combined CSV
    if download_csv and len(all_combined) > 0:
        try:
            combined = pd.concat(all_combined, ignore_index=True)
            st.download_button("Download combined summary CSV", combined.to_csv(index=False).encode('utf-8'), file_name="combined_summary.csv", mime='text/csv')
        except Exception:
            pass

st.write("---")
st.caption("Made with ❤️ — Crypto Market Analyzer. Not financial advice.")
