# app.py
"""
Crypto Market Analyzer - Full single-file app
Place optional header image at images/header.png
Requirements (suggested): streamlit, pandas, numpy, yfinance, plotly, requests, prophet (optional)
"""

import os
import io
import requests
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go

# Optional Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ---------- Page config and header ----------
st.set_page_config(page_title="Crypto Market Analyzer", page_icon="🚀", layout="wide")

st.markdown("""
<style>
.header {display:flex; gap:18px; padding:12px; border-radius:12px; background:linear-gradient(90deg,#0b1220,#07112a); color:#fff}
.logo {width:96px; height:96px; border-radius:12px; background:linear-gradient(135deg,#f6d365,#fda085); display:flex; align-items:center; justify-content:center; font-weight:800; color:#07112a; font-size:26px}
.title {font-size:26px; margin:0; font-weight:800}
.subtitle {font-size:13px; color:#d1d5db; margin-top:6px}
.small {font-size:12px; color:#9ca3af}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 8])
with col1:
    if os.path.exists("images/header.png"):
        st.image("images/header.png", width=110)
    else:
        st.markdown('<div class="logo">CMA</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="header"><div><h1 class="title">Crypto Market Analyzer</h1>'
                '<div class="subtitle">Live prices · Technical indicators · Signals · Forecasting (optional)</div>'
                '<div class="small">Not financial advice — Use responsibly.</div></div></div>', unsafe_allow_html=True)

st.write("---")

# ---------- Helpers ----------
@st.cache_data(ttl=120)
def yf_download_safe(ticker: str, period: str = "3mo", interval: str = "1d"):
    """Download with yfinance; return (df, error_msg)"""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None:
            return pd.DataFrame(), "yfinance returned None"
        return df, None
    except Exception as e:
        return pd.DataFrame(), f"yfinance exception: {e}"

def extract_close(df: pd.DataFrame, prefer_ticker: str = None):
    """Robustly extract Close Series from yfinance df (handles MultiIndex)"""
    if df is None or df.empty:
        return None, "empty dataframe"
    # MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # try level 0 == 'Close'
            lv0 = list(df.columns.get_level_values(0))
            lv1 = list(df.columns.get_level_values(1))
            if 'Close' in lv0:
                cands = [c for c in df.columns if c[0] == 'Close']
                if prefer_ticker:
                    for c in cands:
                        if prefer_ticker in str(c[1]):
                            s = df[c].copy(); s.index = pd.to_datetime(s.index); s.name='Close'; return s.dropna(), ""
                s = df[cands[0]].copy(); s.index = pd.to_datetime(s.index); s.name='Close'; return s.dropna(), ""
            if 'Close' in lv1:
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

def to_numeric_safe(s):
    try:
        return pd.to_numeric(s, errors='coerce')
    except Exception:
        return pd.Series(dtype='float64')

def compute_indicators(series: pd.Series):
    """Return DataFrame with Close, MA20, MA50, MA200, RSI14, MACD_diff"""
    s = to_numeric_safe(series).ffill().dropna()
    if s.empty:
        return pd.DataFrame()
    df = pd.DataFrame({'Close': s})
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(200, min_periods=1).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    ma_down = down.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    df['RSI14'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = macd - signal
    return df

def ensemble_signal(df_ind: pd.DataFrame):
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
        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1
    macd = last.get('MACD_diff', np.nan)
    if not np.isnan(macd):
        score += 1 if macd > 0 else -1
    if score >= 2: return "STRONG BUY"
    if score == 1: return "BUY"
    if score == 0: return "HOLD"
    if score == -1: return "SELL"
    return "STRONG SELL"

def run_prophet(series: pd.Series, days=3):
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet not installed.")
    df = series.dropna().to_frame('y').reset_index().rename(columns={series.index.name or series.name:'ds','y':'y'})
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['ds','y'])
    if df.shape[0] < 2:
        raise ValueError("Not enough data for Prophet.")
    m = Prophet(daily_seasonality=True)
    m.fit(df[['ds','y']])
    future = m.make_future_dataframe(periods=days)
    forecast = m.predict(future)
    return forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(days)

# ---------- Fear & Greed (alternative.me) ----------
@st.cache_data(ttl=600)
def fetch_fear_greed():
    url = "https://api.alternative.me/fng/"
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            j = r.json()
            if 'data' in j and len(j['data'])>0:
                d = j['data'][0]
                val = int(d.get('value',0))
                nice = d.get('value_classification','')
                return {"value":val, "label":nice}
    except Exception:
        pass
    return {"value":55, "label":"Neutral"}

# ---------- News (NewsAPI optional or fallback sample) ----------
SAMPLE_NEWS = [
    {"date":"2025-08-01","title":"Market update: BTC rises","source":"CryptoNews","url":"https://example.com/1"},
    {"date":"2025-07-30","title":"Ethereum upgrade announced","source":"CoinDaily","url":"https://example.com/2"},
]

@st.cache_data(ttl=300)
def fetch_news(news_api_key=None, q="crypto OR bitcoin OR ethereum"):
    if news_api_key:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {"q":q, "pageSize":8, "language":"en", "sortBy":"publishedAt", "apiKey":news_api_key}
            r = requests.get(url, params=params, timeout=6)
            if r.status_code == 200:
                arts = r.json().get('articles',[])
                out = []
                for a in arts:
                    out.append({"date": a.get('publishedAt'), "title": a.get('title'), "source": a.get('source',{}).get('name'), "url": a.get('url')})
                return out
        except Exception:
            pass
    # fallback: return sample
    return SAMPLE_NEWS

# ---------- Sidebar inputs ----------
st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["Manual tickers", "Upload CSV (ds,y)"])
if mode == "Manual tickers":
    default_tickers = "BTC-USD,ETH-USD,ADA-USD,SOL-USD,XRP-USD,DOGE-USD"
    tickers_text = st.sidebar.text_input("Tickers (comma separated)", value=default_tickers)
    tickers_list = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
else:
    uploaded = st.sidebar.file_uploader("Upload CSV (ds,y)", type=["csv"])

history = st.sidebar.selectbox("History", ["1mo","3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)
display_currency = st.sidebar.selectbox("Display currency", ["USD","CAD","GBP","EUR"], index=0)
forecast_days = st.sidebar.selectbox("Forecast days", [3,7,30], index=0)
enable_prophet = st.sidebar.checkbox("Enable Prophet forecasting (optional)", value=PROPHET_AVAILABLE)
if enable_prophet and not PROPHET_AVAILABLE:
    st.sidebar.warning("Prophet not installed. Install or uncheck.")
news_api_key = st.sidebar.text_input("NewsAPI key (optional)", value="")
show_news = st.sidebar.checkbox("Show News & Fear&Greed", value=True)
download_csv = st.sidebar.checkbox("Enable CSV downloads", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: If Prophet not installed, disable forecasting.")

# ---------- Top row: Fear & Greed and quick info ----------
fg = fetch_fear_greed() if show_news else {"value":None,"label":""}
c1, c2, c3 = st.columns([2,3,5])
with c1:
    st.subheader("Fear & Greed")
    if fg.get("value") is not None:
        st.metric("Fear & Greed Index", f"{fg['value']}", fg.get("label",""))
    else:
        st.write("N/A")
with c2:
    st.subheader("Quick Info")
    st.write(f"Display currency: **{display_currency}**")
    st.write(f"History: **{history}** | Interval: **{interval}**")
with c3:
    if show_news:
        st.subheader("Latest news")
        news_items = fetch_news(news_api_key)
        try:
            st.table(pd.DataFrame(news_items))
        except Exception:
            st.write(news_items)

st.write("---")

# ---------- Main: CSV upload or manual tickers ----------
if mode == "Upload CSV (ds,y)":
    if uploaded is None:
        st.info("Upload a CSV with columns `ds` (date) and `y` (price).")
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
            st.subheader("Uploaded Data Indicators")
            st.dataframe(df_ind.tail(10))
            st.markdown(f"**Signal:** {ensemble_signal(df_ind)}")
            if enable_prophet and PROPHET_AVAILABLE:
                try:
                    fc = run_prophet(series, days=forecast_days)
                    fc['ds'] = pd.to_datetime(fc['ds']).dt.date
                    st.subheader(f"Forecast ({forecast_days} days)")
                    st.table(fc.reset_index(drop=True))
                    if download_csv:
                        st.download_button("Download forecast CSV", fc.to_csv(index=False).encode('utf-8'), file_name="uploaded_forecast.csv", mime='text/csv')
                except Exception as e:
                    st.warning(f"Forecast error: {e}")
        else:
            st.error("CSV must include 'ds' and 'y' columns.")
    except Exception as e:
        st.error(f"Failed to parse CSV: {e}")

else:
    st.subheader("Select tickers to analyze")
    sel = st.multiselect("Tickers", options=tickers_list, default=tickers_list[:3])
    manual_ticker = st.text_input("Or add a single ticker manually (e.g., BTC-USD):", value="")
    if manual_ticker.strip():
        if manual_ticker.strip().upper() not in sel:
            sel.append(manual_ticker.strip().upper())

    if not sel:
        st.info("Select at least one ticker.")
        st.stop()

    combined_all = []
    for t in sel:
        st.markdown("---")
        st.markdown(f"### {t}  — display in {display_currency}")
        # download base ticker (t)
        df_raw, err = yf_download_safe(t, period=history, interval=interval)
        if err:
            st.error(f"Error fetching {t}: {err}")
            continue
        # try direct pair with display_currency (e.g., BTC-CAD)
        base = t.split("-")[0] if "-" in t else t
        direct_pair = f"{base}-{display_currency}"
        close_series = None
        dbg = ""
        df_direct, err_direct = yf_download_safe(direct_pair, period=history, interval=interval)
        if err_direct is None and (not df_direct.empty):
            close_series, dbg = extract_close(df_direct, prefer_ticker=direct_pair)
        # fallback to original ticker close (assumed USD)
        if close_series is None:
            close_usd, dbg2 = extract_close(df_raw, prefer_ticker=t)
            if close_usd is None:
                st.error(f"Could not extract Close for {t}. debug: {dbg2}")
                continue
            if display_currency == "USD":
                close_series = close_usd
            else:
                # try to obtain FX rate (USD->display_currency)
                fx_candidates = [f"USD{display_currency}=X", f"{display_currency}=X", f"{display_currency}USD=X", f"US{display_currency}=X"]
                fx_rate = None
                fx_used = None
                for fx in fx_candidates:
                    df_fx, err_fx = yf_download_safe(fx, period="7d", interval="1d")
                    if err_fx is None and not df_fx.empty:
                        sfx, _ = extract_close(df_fx)
                        if sfx is not None and not sfx.empty:
                            try:
                                fx_rate = float(sfx.iloc[-1])
                                fx_used = fx
                                break
                            except Exception:
                                continue
                if fx_rate is None:
                    st.info(f"No FX pair found to convert to {display_currency}; showing USD prices for {t}.")
                    close_series = close_usd
                else:
                    # Heuristic on orientation: if fx ticker startswith USD assume 1 USD = fx_rate CUR
                    if fx_used.upper().startswith("USD") or fx_used.upper().startswith("US"):
                        close_series = close_usd * fx_rate
                    else:
                        # assume fx is CUR per USD? best-effort:
                        try:
                            close_series = close_usd * fx_rate
                        except Exception:
                            close_series = close_usd
                    st.info(f"Applied FX {fx_used} rate to convert to {display_currency}.")
        # ensure we have series
        if close_series is None or close_series.dropna().empty:
            st.error(f"No close series for {t}. {dbg}")
            continue

        # compute indicators
        df_ind = compute_indicators(close_series)
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
        sig = ensemble_signal(df_ind)
        st.markdown(f"**Signal:** `{sig}`")

        # table preview
        st.dataframe(df_ind.tail(8).reset_index().rename(columns={df_ind.index.name or 'index':'Date'}))

        # forecast (if enabled)
        forecast_df = None
        if enable_prophet and PROPHET_AVAILABLE:
            try:
                forecast_df = run_prophet(close_series, days=forecast_days)
            except Exception as e:
                st.warning(f"Prophet forecast error: {e}")
                forecast_df = None

        # main plot
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['Close'], name='Close'))
            if 'MA20' in df_ind.columns: fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MA20'], name='MA20'))
            if 'MA50' in df_ind.columns: fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind['MA50'], name='MA50'))
            if forecast_df is not None:
                fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast', line=dict(dash='dash')))
                fig.add_trace(go.Scatter(x=pd.concat([forecast_df['ds'], forecast_df['ds'][::-1]]),
                                         y=pd.concat([forecast_df['yhat_upper'], forecast_df['yhat_lower'][::-1]]),
                                         fill='toself', fillcolor='rgba(214,39,40,0.08)', line=dict(color='rgba(255,255,255,0)'), hoverinfo='skip', showlegend=False))
            fig.update_layout(title=f"{t} price ({display_currency})", xaxis_rangeslider_visible=True, height=480)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plot error: {e}")

        # RSI and MACD plots
        try:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_ind.index, y=df_ind['RSI14'], name='RSI14'))
            fig_rsi.update_layout(title='RSI (14)', yaxis=dict(range=[0,100]), height=260)
            st.plotly_chart(fig_rsi, use_container_width=True)
        except Exception:
            pass
        try:
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Bar(x=df_ind.index, y=df_ind['MACD_diff'], name='MACD diff'))
            fig_macd.update_layout(title='MACD diff', height=260)
            st.plotly_chart(fig_macd, use_container_width=True)
        except Exception:
            pass

        # downloads
        if download_csv:
            try:
                st.download_button(f"Download {t} indicators CSV", df_ind.to_csv().encode('utf-8'), file_name=f"{t}_indicators.csv", mime='text/csv')
            except Exception:
                pass
        if forecast_df is not None and download_csv:
            try:
                st.download_button(f"Download {t} forecast CSV", forecast_df.to_csv(index=False).encode('utf-8'), file_name=f"{t}_forecast.csv", mime='text/csv')
            except Exception:
                pass

        # collect combined
        tail = df_ind.tail(5).reset_index().rename(columns={df_ind.index.name or 'index':'Date'})
        tail['ticker'] = t
        combined = tail[['ticker','Date','Close','MA20','MA50','RSI14','MACD_diff']].copy()
        combined_all.append(combined)

    # combined download
    if download_csv and len(combined_all)>0:
        try:
            combined_df = pd.concat(combined_all, ignore_index=True)
            st.download_button("Download combined summary CSV", combined_df.to_csv(index=False).encode('utf-8'), file_name="combined_summary.csv", mime='text/csv')
        except Exception:
            pass

st.write("---")
st.caption("Made with ❤️ — Crypto Market Analyzer. Not financial advice.")
