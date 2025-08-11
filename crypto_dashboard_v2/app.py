# app.py
"""
Golden Market Analyzer - Dark + Gold (Streamlit)
English UI. Prophet removed for lighter package.
"""
import os
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests

st.set_page_config(page_title="Golden Market Analyzer", page_icon="💰", layout="wide")

# -------------------------
# Styling (dark + gold)
# -------------------------
st.markdown("""
<style>
:root { --gold1: #d4af37; --gold2: #ffd27f; --bg: #0b0b0b; --card: #0f1724; --muted: #9ca3af; }
.stApp { background: var(--bg); color: #ffffff; }
.header { display:flex; gap:16px; align-items:center; padding:8px; border-radius:8px; background: linear-gradient(90deg,#080808,#0f1112); }
.title { font-size:28px; font-weight:800; color:var(--gold1); margin:0; }
.subtitle { color: #cbd5e1; margin:0; font-size:13px; }
.small { color: var(--muted); font-size:12px; margin-top:6px; }
.card { background: var(--card); padding:10px; border-radius:8px; border:1px solid rgba(212,175,55,0.06); }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header image or fallback
# -------------------------
col1, col2 = st.columns([1, 8])
with col1:
    header_path = os.path.join("assets", "header.png")
    if os.path.exists(header_path):
        st.image(header_path, use_column_width=False, width=140)
    else:
        st.markdown('<div style="width:140px;height:80px;border-radius:8px;background:linear-gradient(135deg,#d4af37,#b8860b);display:flex;align-items:center;justify-content:center;font-weight:800;color:#07112a">GMA</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="header"><div><h1 class="title">Golden Market Analyzer</h1><div class="subtitle">Live prices · Indicators · Signals · News</div><div class="small">Not financial advice — Use responsibly.</div></div></div>', unsafe_allow_html=True)

st.write("---")

# -------------------------
# Helpers
# -------------------------
@st.cache_data(ttl=120)
def safe_yf_download(ticker: str, period: str = "3mo", interval: str = "1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
        if df is None:
            return pd.DataFrame(), "yfinance returned None"
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

def extract_close(df: pd.DataFrame):
    if df is None or df.empty:
        return None, "empty dataframe"
    cols = df.columns
    if isinstance(cols, pd.MultiIndex):
        # try find any level containing 'close'
        for col in cols:
            try:
                if any("close" in str(item).lower() for item in col if item is not None):
                    s = df[col].copy(); s.index = pd.to_datetime(s.index); s.name = "Close"; return s.dropna(), ""
            except Exception:
                continue
        # fallback: first numeric column
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                s = df[col].copy(); s.index = pd.to_datetime(s.index); s.name = "Close"; return s.dropna(), ""
        return None, "multiindex but no close-like column"
    # non-multiindex
    for cand in ["Close", "close", "Adj Close", "Adj_Close", "AdjClose"]:
        if cand in df.columns:
            s = df[cand].copy(); s.index = pd.to_datetime(s.index); s.name = "Close"; return s.dropna(), ""
    # fallback
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        s = df[numeric_cols[0]].copy(); s.index = pd.to_datetime(s.index); s.name = "Close"; return s.dropna(), ""
    return None, "no numeric close column"

def to_numeric_series(s):
    try:
        out = pd.to_numeric(s, errors="coerce").astype(float)
        out.index = pd.to_datetime(out.index)
        return out.dropna()
    except Exception:
        try:
            arr = np.array(list(s))
            return pd.Series(pd.to_numeric(arr, errors="coerce")).dropna()
        except Exception:
            return pd.Series(dtype="float64")

def compute_indicators(series: pd.Series):
    s = to_numeric_series(series).ffill().dropna()
    if s.empty:
        return pd.DataFrame()
    df = pd.DataFrame({"Close": s})
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["MA200"] = df["Close"].rolling(200, min_periods=1).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["MACD_diff"] = macd - signal
    return df

def ensemble_signal(df_ind: pd.DataFrame):
    if df_ind.empty:
        return "NO DATA"
    last = df_ind.iloc[-1]
    score = 0
    try:
        if last["Close"] > last.get("MA20", 0) > last.get("MA50", 0):
            score += 1
        elif last["Close"] < last.get("MA20", 0) < last.get("MA50", 0):
            score -= 1
    except Exception:
        pass
    rsi = last.get("RSI14", np.nan)
    if not np.isnan(rsi):
        if rsi < 30:
            score += 1
        elif rsi > 70:
            score -= 1
    macd = last.get("MACD_diff", np.nan)
    if not np.isnan(macd):
        score += 1 if macd > 0 else -1
    if score >= 2: return "STRONG BUY"
    if score == 1: return "BUY"
    if score == 0: return "HOLD"
    if score == -1: return "SELL"
    return "STRONG SELL"

# -------------------------
# Fear & Greed (alternative.me)
# -------------------------
@st.cache_data(ttl=600)
def fetch_fng():
    url = "https://api.alternative.me/fng/"
    try:
        r = requests.get(url, timeout=5)
        if r.ok:
            j = r.json()
            d = j.get("data", [{}])[0]
            return {"value": int(d.get("value", 50)), "label": d.get("value_classification", "Neutral")}
    except Exception:
        pass
    return {"value": None, "label": "Unknown"}

# -------------------------
# News sample / optional NewsAPI
# -------------------------
SAMPLE_NEWS = [
    {"date": "2025-08-01", "title": "Market update: BTC rises", "source": "CryptoNews", "url": "https://example.com/1"},
    {"date": "2025-07-30", "title": "Ethereum upgrade announced", "source": "CoinDaily", "url": "https://example.com/2"},
]

@st.cache_data(ttl=300)
def fetch_news(api_key=None):
    if api_key:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {"q": "crypto OR bitcoin OR ethereum", "pageSize": 6, "language": "en", "sortBy": "publishedAt", "apiKey": api_key}
            r = requests.get(url, params=params, timeout=6)
            if r.ok:
                arts = r.json().get("articles", [])
                return [{"date": a.get("publishedAt"), "title": a.get("title"), "source": a.get("source", {}).get("name"), "url": a.get("url")} for a in arts]
        except Exception:
            pass
    return SAMPLE_NEWS

# -------------------------
# Sidebar settings
# -------------------------
st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["Manual tickers", "Upload CSV"])
if mode == "Manual tickers":
    default = "BTC-USD,ETH-USD,ADA-USD,SOL-USD,XRP-USD"
    tickers_text = st.sidebar.text_input("Tickers (comma separated)", value=default)
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
else:
    uploaded = st.sidebar.file_uploader("Upload CSV (ds,y)", type=["csv"])

history = st.sidebar.selectbox("History", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)
display_currency = st.sidebar.selectbox("Display currency", ["USD", "CAD", "GBP", "EUR"], index=0)
show_news = st.sidebar.checkbox("Show News & Fear&Greed", value=True)
enable_downloads = st.sidebar.checkbox("Enable CSV downloads", value=True)
news_api_key = st.sidebar.text_input("NewsAPI key (optional)", value="")

# -------------------------
# Top row (FNG + quick info + news)
# -------------------------
fg = fetch_fng() if show_news else {"value": None, "label": ""}
c1, c2, c3 = st.columns([2, 3, 5])
with c1:
    st.subheader("Fear & Greed")
    if fg.get("value") is not None:
        st.metric("Index", f"{fg['value']}", fg.get("label", ""))
        pct = max(0, min(100, fg["value"]))
        st.markdown(f'<div style="background:#222;border-radius:8px;padding:3px"><div style="width:{pct}%;background:linear-gradient(90deg,#d4af37,#ffd27f);height:18px;border-radius:6px"></div></div>', unsafe_allow_html=True)
    else:
        st.write("N/A")
with c2:
    st.subheader("Quick Info")
    st.write(f"Currency: **{display_currency}**")
    st.write(f"History: **{history}** | Interval: **{interval}**")
with c3:
    if show_news:
        st.subheader("News")
        news_list = fetch_news(news_api_key)
        try:
            st.table(pd.DataFrame(news_list))
        except Exception:
            st.write(news_list)

st.write("---")

# -------------------------
# Main
# -------------------------
if mode == "Upload CSV":
    if uploaded is None:
        st.info("Please upload a CSV with columns `ds` and `y` (dates and numeric values).")
        st.stop()
    df_csv = pd.read_csv(uploaded)
    cols = {c.lower(): c for c in df_csv.columns}
    if "ds" in cols and "y" in cols:
        ds_col = cols["ds"]; y_col = cols["y"]
        df_csv[ds_col] = pd.to_datetime(df_csv[ds_col], errors="coerce")
        df_csv[y_col] = pd.to_numeric(df_csv[y_col], errors="coerce")
        df_csv = df_csv.dropna(subset=[ds_col, y_col]).set_index(ds_col).sort_index()
        series = df_csv[y_col]
        df_ind = compute_indicators(series)
        st.subheader("Uploaded Data Indicators (tail)")
        st.dataframe(df_ind.tail(10))
        st.markdown(f"**Signal:** {ensemble_signal(df_ind)}")
    else:
        st.error("CSV must contain columns named `ds` and `y`.")
else:
    st.subheader("Select Tickers")
    selected = st.multiselect("Tickers", options=tickers, default=tickers[:3])
    add_manual = st.text_input("Add ticker manually (e.g. BTC-USD):", value="")
    if add_manual.strip():
        if add_manual.strip().upper() not in selected:
            selected.append(add_manual.strip().upper())

    if not selected:
        st.info("Select at least one ticker to continue.")
        st.stop()

    summary_rows = []
    for ticker in selected:
        st.markdown("---")
        st.markdown(f"### {ticker}  —  {display_currency}")
        df_raw, err = safe_yf_download(ticker, period=history, interval=interval)
        if err:
            st.error(f"Download error for {ticker}: {err}")
            continue
        series_close, dbg = extract_close(df_raw)
        if series_close is None:
            st.error(f"No Close series for {ticker}: {dbg}")
            continue

        # currency conversion best-effort
        if display_currency != "USD":
            FxCandidates = [f"USD{display_currency}=X", f"{display_currency}=X"]
            fx_val = None
            for fx in FxCandidates:
                fx_df, fx_err = safe_yf_download(fx, period="7d", interval="1d")
                if fx_err is None and not fx_df.empty:
                    sfx, _ = extract_close(fx_df)
                    if sfx is not None and not sfx.empty:
                        try:
                            fx_val = float(sfx.iloc[-1])
                            break
                        except Exception:
                            continue
            if fx_val is not None:
                series_for_calc = series_close * fx_val
            else:
                series_for_calc = series_close
                st.info(f"No FX pair found for {display_currency}; showing USD values.")
        else:
            series_for_calc = series_close

        df_ind = compute_indicators(series_for_calc)
        if df_ind.empty:
            st.error("Not enough numeric data to compute indicators.")
            continue

        last = df_ind.iloc[-1]
        last_price = float(last["Close"])
        # smart formatting
        if last_price >= 1:
            price_str = f"${last_price:,.2f}"
        else:
            price_str = f"${last_price:,.6f}"
        st.metric(f"{ticker} Last ({display_currency})", price_str)
        st.markdown(f"**Signal:** {ensemble_signal(df_ind)}")

        st.dataframe(df_ind.tail(8).reset_index().rename(columns={df_ind.index.name or "index":"Date"}))

        # Plot
        try:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["Close"], name="Close", line=dict(color="#FFFFFF")))
            if "MA20" in df_ind.columns: fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MA20"], name="MA20", line=dict(color="#FFD27F")))
            if "MA50" in df_ind.columns: fig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["MA50"], name="MA50", line=dict(color="#D4AF37")))
            fig.update_layout(template="plotly_dark", height=480, xaxis_rangeslider_visible=True, title=f"{ticker} ({display_currency})")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Plot error: {e}")

        # RSI & MACD small charts
        try:
            rfig = go.Figure(); rfig.add_trace(go.Scatter(x=df_ind.index, y=df_ind["RSI14"], name="RSI14", line=dict(color="#FFD27F"))); rfig.update_layout(template="plotly_dark", height=220, title="RSI (14)"); st.plotly_chart(rfig, use_container_width=True)
        except Exception:
            pass
        try:
            mfig = go.Figure(); mfig.add_trace(go.Bar(x=df_ind.index, y=df_ind["MACD_diff"], name="MACD diff", marker_color="#D4AF37")); mfig.update_layout(template="plotly_dark", height=220, title="MACD diff"); st.plotly_chart(mfig, use_container_width=True)
        except Exception:
            pass

        if enable_downloads:
            try: st.download_button(f"Download {ticker} indicators (CSV)", df_ind.to_csv().encode("utf-8"), file_name=f"{ticker}_indicators.csv", mime="text/csv")
            except Exception: pass

        tail = df_ind.tail(1).reset_index().rename(columns={df_ind.index.name or "index":"Date"})
        tail["ticker"] = ticker
        summary_rows.append(tail[["ticker","Date","Close","MA20","MA50","RSI14","MACD_diff"]])

    if enable_downloads and summary_rows:
        try:
            combined = pd.concat(summary_rows, ignore_index=True)
            st.download_button("Download combined summary CSV", combined.to_csv(index=False).encode("utf-8"), file_name="combined_summary.csv", mime="text/csv")
        except Exception:
            pass

st.write("---")
st.caption("Made with ❤️ — Golden Market Analyzer. Not financial advice.")
