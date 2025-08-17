# app.py
# Global Crypto Insight — Dual-source (Yahoo -> CoinGecko) — gold+black theme
# Single-file Streamlit app. Prophet & ta optional.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta, timezone

st.set_page_config(page_title="Global Crypto Insight", page_icon="🟡", layout="wide")
pd.options.mode.chained_assignment = None

# ---------- Optional libs ----------
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

# ---------- Helpers (format/indicators) ----------
def ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df

def normalize_ohlc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = ensure_flat_columns(df).reset_index()
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    keep = ["Open","High","Low","Close","Adj Close","Volume"]
    cols = ["Date"] + [c for c in keep if c in df.columns]
    df = df[cols].dropna(subset=["Date"]).sort_values("Date")
    return df

def series_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float, name="Close")
    col = None
    if "Close" in df.columns:
        col = "Close"
    else:
        try:
            m = [c for c in df.columns if str(c).lower() == "close"]
            col = m[0] if m else None
        except Exception:
            col = None
    if col is None:
        return pd.Series(dtype=float, name="Close")
    s = pd.to_numeric(df[col], errors="coerce")
    s.name = "Close"
    return s

def moving_avg_forecast(close_s: pd.Series, days: int) -> np.ndarray:
    try:
        s = pd.to_numeric(close_s, errors="coerce").dropna()
        if s.empty:
            return np.array([np.nan]*days)
        last = float(s.iloc[-1])
        mean_pct = s.pct_change().dropna().mean() if s.shape[0] > 1 else 0.0
        if not np.isfinite(mean_pct):
            mean_pct = 0.0
        return np.array([ last * ((1 + mean_pct) ** i) for i in range(1, days+1) ])
    except Exception:
        return np.array([np.nan]*days)

def simple_rsi(series, window=14):
    series = pd.to_numeric(series, errors="coerce")
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1.0 * delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def simple_macd(series, fast=12, slow=26, signal=9):
    series = pd.to_numeric(series, errors="coerce")
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff

def compute_combined_signal(close_series, next_forecast=None):
    cs = pd.to_numeric(close_series, errors="coerce").dropna()
    if cs.empty:
        return ("NO DATA", "#9e9e9e", "insufficient data")
    ma20 = cs.rolling(20, min_periods=1).mean().iloc[-1]
    ma50 = cs.rolling(50, min_periods=1).mean().iloc[-1]
    ma200 = cs.rolling(200, min_periods=1).mean().iloc[-1]

    try:
        if HAS_TA:
            rsi = ta.momentum.RSIIndicator(cs, window=14).rsi().iloc[-1]
            macd_diff = ta.trend.MACD(cs).macd_diff().iloc[-1]
        else:
            rsi = simple_rsi(cs).iloc[-1]
            _, _, md = simple_macd(cs)
            macd_diff = md.iloc[-1]
    except Exception:
        rsi = np.nan
        macd_diff = np.nan

    score, reasons = 0, []
    # Trend (MAs)
    if ma20 > ma50:
        score += 2; reasons.append("MA20>MA50")
    else:
        score -= 1; reasons.append("MA20<=MA50")
    if np.isfinite(ma200):
        if ma50 > ma200:
            score += 1; reasons.append("MA50>MA200")
        else:
            score -= 1; reasons.append("MA50<=MA200")
    # RSI
    if pd.notna(rsi):
        if rsi < 30: score += 1; reasons.append(f"RSI low ({rsi:.1f})")
        elif rsi > 70: score -= 1; reasons.append(f"RSI high ({rsi:.1f})")
    # MACD
    if pd.notna(macd_diff):
        if macd_diff > 0: score += 1; reasons.append("MACD+")
        else: score -= 1; reasons.append("MACD-")
    # Forecast tilt
    if next_forecast is not None and np.isfinite(next_forecast):
        last = float(cs.iloc[-1])
        if last != 0:
            pct = (next_forecast - last) / last
            if pct > 0.01: score += 1; reasons.append(f"Forecast +{pct*100:.2f}%")
            elif pct < -0.01: score -= 1; reasons.append(f"Forecast {pct*100:.2f}%")

    if score >= 4: return ("STRONG BUY","#d4ffb3"," · ".join(reasons))
    if score >= 2: return ("BUY","#b2ff66"," · ".join(reasons))
    if score == 1: return ("MILD BUY","#ffe36b"," · ".join(reasons))
    if score == 0: return ("HOLD","#cfd8dc"," · ".join(reasons))
    if score == -1: return ("MILD SELL","#ffb86b"," · ".join(reasons))
    return ("SELL","#ff7b7b"," · ".join(reasons))

def fmt_currency(x, cur="USD"):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "—"
        return f"{x:,.2f} {cur}"
    except Exception:
        return "—"

# ---------- CoinGecko utilities (fallback data) ----------
COINGECKO_STATIC_MAP = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "BNB-USD": "binancecoin",
    "SOL-USD": "solana",
    "XRP-USD": "ripple",
    "ADA-USD": "cardano",
    "DOGE-USD": "dogecoin",
    "TON-USD": "the-open-network",
    "TRX-USD": "tron",
    "AVAX-USD": "avalanche-2",
    "SHIB-USD": "shiba-inu",
    "DOT-USD": "polkadot",
    "LINK-USD": "chainlink",
    "LTC-USD": "litecoin",
    "BCH-USD": "bitcoin-cash",
    "NEAR-USD": "near",
    "UNI-USD": "uniswap",
    "MATIC-USD": "matic-network",
    "ETC-USD": "ethereum-classic",
    "ATOM-USD": "cosmos",
    "ENA-USD": "ethena",          # ENA
    "USDT-USD": "tether",         # (stable)
    "USDC-USD": "usd-coin",       # (stable)
}

def _symbol_to_coingecko_id(symbol: str) -> str | None:
    symbol = symbol.upper().strip()
    if symbol in COINGECKO_STATIC_MAP:
        return COINGECKO_STATIC_MAP[symbol]
    # Try dynamic lookup: /coins/list
    try:
        r = requests.get("https://api.coingecko.com/api/v3/coins/list", timeout=10)
        if r.status_code == 200:
            items = r.json()
            # Try exact by symbol (without -USD)
            sym = symbol.replace("-USD","").lower()
            matches = [x for x in items if x.get("symbol","").lower() == sym]
            if matches:
                # prefer the one with matching name or top result
                return matches[0]["id"]
            # Try by name (rare)
            matches = [x for x in items if x.get("name","").lower() == sym]
            if matches:
                return matches[0]["id"]
    except Exception:
        pass
    return None

def _period_to_days(period: str) -> int:
    mp = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}
    return mp.get(period, 90)

def _cg_build_ohlc_from_prices(prices: list, freq: str = "D") -> pd.DataFrame:
    # prices: [[ms, price], ...]
    if not prices:
        return pd.DataFrame()
    df = pd.DataFrame(prices, columns=["ts", "price"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(timezone.utc).dt.tz_localize(None)
    df = df[["Date","price"]]
    # resample to daily or hourly OHLC
    df = df.set_index("Date").sort_index()
    rule = "H" if freq == "1h" else "D"
    ohlc = df["price"].resample(rule).agg(["first","max","min","last"])
    ohlc.columns = ["Open","High","Low","Close"]
    ohlc["Volume"] = np.nan
    ohlc["Adj Close"] = ohlc["Close"]
    ohlc = ohlc.reset_index().rename(columns={"index":"Date"})
    return ohlc

def fetch_coingecko(symbol: str, period="3mo", interval="1d") -> pd.DataFrame:
    coin_id = _symbol_to_coingecko_id(symbol)
    if not coin_id:
        return pd.DataFrame()
    days = _period_to_days(period)
    vs = "usd"  # نمایش بر اساس USD، تبدیل ارزی پایین اعمال می‌شود
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": vs, "days": days, "interval": "hourly" if interval=="1h" else "daily"}
    try:
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return pd.DataFrame()
        j = r.json()
        prices = j.get("prices", [])
        df = _cg_build_ohlc_from_prices(prices, freq="1h" if interval=="1h" else "1d")
        return df
    except Exception:
        return pd.DataFrame()

# ---------- Data fetch (Yahoo -> CoinGecko) ----------
@st.cache_data(ttl=180)
def fetch_yf(symbol: str, period="3mo", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return normalize_ohlc_index(df)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=180)
def fetch_data(symbol: str, period="3mo", interval="1d") -> tuple[pd.DataFrame, str]:
    """Returns (df, source) where source in {'yahoo','coingecko','none'}."""
    df = fetch_yf(symbol, period=period, interval=interval)
    if df is not None and not df.empty:
        return (df, "yahoo")
    df2 = fetch_coingecko(symbol, period=period, interval=interval)
    if df2 is not None and not df2.empty:
        return (df2, "coingecko")
    return (pd.DataFrame(), "none")

# ---------- Fear & Greed ----------
@st.cache_data(ttl=600)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        if r.status_code == 200:
            j = r.json()
            if "data" in j and len(j["data"])>0:
                e = j["data"][0]
                return {
                    "value": int(e.get("value",50)),
                    "class": e.get("value_classification","Neutral"),
                    "date": datetime.utcfromtimestamp(int(e.get("timestamp",0))).strftime("%Y-%m-%d")
                            if e.get("timestamp") else None
                }
    except Exception:
        pass
    return {"value": None, "class":"N/A", "date":None}

# ---------- FX (USD -> target) ----------
@st.cache_data(ttl=300)
def get_fx(target):
    if target == "USD": return 1.0
    mp = {"CAD": "USDCAD=X", "EUR": "USDEUR=X", "GBP": "USDGBP=X"}
    t = mp.get(target)
    if not t: return 1.0
    try:
        df = yf.download(t, period="5d", interval="1d", progress=False)
        if df is None or df.empty: return 1.0
        df = normalize_ohlc_index(df)
        s = series_close(df).dropna()
        if s.empty: return 1.0
        return float(s.iloc[-1])
    except Exception:
        return 1.0

# ---------- UI Header ----------
st.markdown(
    """
    <style>
      .gci-header{padding:20px;border-radius:12px;
        background: linear-gradient(90deg,#0b0b0b 0%, #1a1a1a 50%, #3a2b10 100%);
        color: #f9f4ef; box-shadow: 0 10px 30px rgba(0,0,0,0.5);}
      .gci-title{font-size:26px;font-weight:800;color:#f5d76e}
      .gci-sub{color:#e6dccd;margin-top:6px}
    </style>
    <div class="gci-header">
      <div class="gci-title">Global Crypto Insight</div>
      <div class="gci-sub">Live market · Dual-source data · Forecasts · Signals · Heatmap · Portfolio</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
st.sidebar.header("Settings")

DEFAULTS = ["BTC-USD","ETH-USD","ADA-USD","SOL-USD","ENA-USD"]
currency = st.sidebar.selectbox("Display currency", ["USD","CAD","EUR","GBP"], index=0)
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)

primary = st.sidebar.selectbox("Primary symbol", DEFAULTS, index=0)
extra_text = st.sidebar.text_area("Extra symbols (comma separated)", value="XRP-USD,DOGE-USD,BNB-USD")
extra_list = [s.strip().upper() for s in extra_text.split(",") if s.strip()]

# Build symbol list (dedup, keep order)
symbols = []
for s in [primary] + [x for x in DEFAULTS if x != primary] + extra_list:
    if s not in symbols:
        symbols.append(s)

st.sidebar.markdown("---")
st.sidebar.caption("Prophet & ta اختیاری هستند. نصب‌شان، پیش‌بینی و اندیکاتورها را ارتقا می‌دهد.")

fx = get_fx(currency)

# ---------- Tabs ----------
tabs = st.tabs(["Market","Forecast","Portfolio","News","About"])
tab_market, tab_forecast, tab_portfolio, tab_news, tab_about = tabs

# ---------- Market Tab ----------
with tab_market:
    st.header("Market Overview")

    fg = fetch_fear_greed()
    c1,c2,c3,c4 = st.columns([1.2,2,2,2])

    with c1:
        if fg['value'] is not None:
            figg = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fg['value'],
                title={"text": f"Fear & Greed ({fg['date']})"},
                gauge={"axis":{"range":[0,100]},
                       "bar":{"color":"#f5d76e"},
                       "steps":[
                           {"range":[0,25],"color":"#5c1f1f"},
                           {"range":[25,40],"color":"#9b5f00"},
                           {"range":[40,60],"color":"#b0894a"},
                           {"range":[60,75],"color":"#7bb383"},
                           {"range":[75,100],"color":"#2b9348"}]}))
            figg.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figg, use_container_width=True)
        else:
            st.info("Fear & Greed: N/A")

    with c2:
        st.markdown("### Snapshot")
        s0 = primary
        d0, src0 = fetch_data(s0, period=period, interval=interval)
        c0 = series_close(d0).dropna()
        if not c0.empty:
            last = float(c0.iloc[-1]) * fx
            prev = float(c0.iloc[-2]) * fx if c0.shape[0] >= 2 else last
            ch = (last - prev)/prev*100 if prev!=0 else 0.0
            st.metric(f"{s0} Latest ({currency})", f"{last:,.2f}", delta=f"{ch:+.2f}%")
            st.caption(f"Data source: {src0}")
        else:
            st.info(f"{s0}: no data")

    with c3:
        st.markdown("### Period High")
        rows=[]
        for s in symbols[:6]:
            d, _ = fetch_data(s, period=period, interval=interval)
            if not d.empty:
                # اگر High نبود، از Close بیشینه می‌گیریم
                if "High" in d.columns and d["High"].notna().any():
                    hi = pd.to_numeric(d["High"], errors="coerce").dropna()
                    if not hi.empty:
                        rows.append(f"{s}: {float(hi.max())*fx:,.2f} {currency}")
                else:
                    c = series_close(d).dropna()
                    if not c.empty:
                        rows.append(f"{s}: {float(c.max())*fx:,.2f} {currency}")
        if rows: st.markdown("<br/>".join(rows), unsafe_allow_html=True)
        else: st.info("No high data.")

    with c4:
        st.markdown("### Period Low")
        rows=[]
        for s in symbols[:6]:
            d, _ = fetch_data(s, period=period, interval=interval)
            if not d.empty:
                if "Low" in d.columns and d["Low"].notna().any():
                    lo = pd.to_numeric(d["Low"], errors="coerce").dropna()
                    if not lo.empty:
                        rows.append(f"{s}: {float(lo.min())*fx:,.2f} {currency}")
                else:
                    c = series_close(d).dropna()
                    if not c.empty:
                        rows.append(f"{s}: {float(c.min())*fx:,.2f} {currency}")
        if rows: st.markdown("<br/>".join(rows), unsafe_allow_html=True)
        else: st.info("No low data.")

    st.markdown("---")
    # Summary table
    summary=[]
    with st.spinner("Fetching market data..."):
        for s in symbols[:20]:
            d, src = fetch_data(s, period=period, interval=interval)
            c = series_close(d).dropna()
            if c.empty:
                summary.append({"Symbol":s,"Price":None,"Change24h":None,"Signal":"NO DATA","Color":"#9e9e9e","Reason":"—","Source":src})
                continue
            price_usd = float(c.iloc[-1])
            price = price_usd * fx
            prev_usd = float(c.iloc[-2]) if c.shape[0] >= 2 else price_usd
            change24 = (price_usd - prev_usd)/prev_usd*100 if prev_usd!=0 else 0.0
            fc1 = moving_avg_forecast(c, 1)
            next_fc = float(fc1[0]) if len(fc1)>0 else None
            label,color,reason = compute_combined_signal(c, next_fc)
            summary.append({
                "Symbol": s,
                "Price": price,
                "Change24h": round(change24,2) if np.isfinite(change24) else None,
                "Signal": label,
                "Color": color,
                "Reason": reason,
                "Source": src
            })
    df_sum = pd.DataFrame(summary)
    if not df_sum.empty:
        df_sum["PriceStr"] = df_sum["Price"].apply(lambda v: fmt_currency(v, currency))
        df_sum["ChangeStr"] = df_sum["Change24h"].apply(lambda v: "—" if (v is None or (isinstance(v,float) and not np.isfinite(v))) else f"{v:+.2f}%")
        st.dataframe(
            df_sum[["Symbol","PriceStr","ChangeStr","Signal","Source"]],
            use_container_width=True, hide_index=True
        )
    else:
        st.info("No data.")

    st.markdown("### Signal Cards")
    ncols = min(6, max(1, len(df_sum)))
    cols = st.columns(ncols) if ncols>0 else []
    for i,row in df_sum.iterrows():
        c = cols[i % ncols]
        with c:
            html = f"""
            <div style='background:{row["Color"]};padding:10px;border-radius:8px;text-align:center;color:#021014;'>
              <strong>{row["Symbol"]}</strong><br/>{row["Signal"]}<br/>{row["PriceStr"]} · {row["ChangeStr"]}
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Trend Heatmap (daily %)")
    heat_days = st.slider("Heatmap lookback (days)", min_value=5, max_value=30, value=14)
    heat_data = {}
    dates_idx = None
    for s in symbols[:20]:
        d, _ = fetch_data(s, period="2mo", interval="1d")
        if d.empty or "Date" not in d.columns:
            continue
        cs = series_close(d)
        if cs.dropna().empty:
            continue
        ser = d.set_index("Date")[cs.name].resample("D").ffill().dropna()
        ser = ser.tail(heat_days+1)
        if ser.shape[0] < 2:
            continue
        rets = ser.pct_change().dropna()*100
        heat_data[s] = rets
        dates_idx = rets.index if dates_idx is None else dates_idx.intersection(rets.index)
    if heat_data and dates_idx is not None and len(dates_idx) > 0:
        heat_df = pd.DataFrame(heat_data).loc[dates_idx].T
        if heat_df.shape[1] > 0:
            lastcol = heat_df.columns[-1]
            heat_df = heat_df.reindex(heat_df[lastcol].sort_values(ascending=False).index)
        fig_h = go.Figure(data=go.Heatmap(
            z=np.round(heat_df.values,2),
            x=[d.strftime("%Y-%m-%d") for d in heat_df.columns],
            y=heat_df.index,
            colorscale="RdYlGn"
        ))
        fig_h.update_layout(height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Not enough data for heatmap.")

# ---------- Forecast Tab ----------
with tab_forecast:
    st.header("Forecast & Backtest")
    f_sym = st.selectbox("Choose symbol", options=symbols[:20], index=0 if symbols else None)
    f_period = st.selectbox("History period", ["3mo","6mo","1y","2y"], index=0)
    f_interval = st.selectbox("Interval", ["1d","1h"], index=0)
    f_model = st.selectbox("Model", ["Prophet (if installed)","MovingAvg (fallback)"], index=1)
    f_horizon = st.selectbox("Horizon (days)", [3,7,14,30], index=1)

    if f_sym:
        df_f, src_f = fetch_data(f_sym, period=f_period, interval=f_interval)
        cs = series_close(df_f).dropna()
        if df_f.empty or cs.empty:
            st.warning("No historical data.")
        else:
            st.caption(f"Data source: {src_f}")
            st.subheader(f"Historical: {f_sym}")
            st.line_chart(df_f.set_index("Date")[cs.name])

            # Forecast
            with st.spinner("Running forecast..."):
                try:
                    if f_model.startswith("Prophet") and HAS_PROPHET and df_f.shape[0] >= 30:
                        pf = df_f[["Date", cs.name]].rename(columns={"Date":"ds", cs.name:"y"})
                        m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
                        m.fit(pf)
                        future = m.make_future_dataframe(periods=f_horizon, freq="D")
                        pred = m.predict(future)
                        tail = pred.tail(f_horizon)
                        forecast_vals = tail["yhat"].values
                        fc_dates = tail["ds"].dt.date.values
                    else:
                        arr = moving_avg_forecast(cs, f_horizon)
                        forecast_vals = arr
                        last_date = df_f["Date"].iloc[-1]
                        fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]
                except Exception:
                    arr = moving_avg_forecast(cs, f_horizon)
                    forecast_vals = arr
                    last_date = df_f["Date"].iloc[-1]
                    fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]

            fc_table = pd.DataFrame({"Date": fc_dates, "Predicted": np.round(forecast_vals, 6)})
            st.subheader("Forecast")
            st.table(fc_table)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_f["Date"], y=cs, mode="lines", name="History"))
            fig.add_trace(go.Scatter(x=fc_dates, y=forecast_vals, mode="lines+markers",
                                     name=f"Forecast {f_horizon}d",
                                     line=dict(dash="dash", color="#f5d76e")))
            fig.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(fig, use_container_width=True)

            # Backtest (simple, horizon=1)
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            def backtest_simple(df_hist, colname="Close", lookback_days=90):
                df2 = df_hist[["Date", colname]].set_index("Date").asfreq("D", method="pad").dropna()
                n = len(df2); preds=[]; truths=[]
                for i in range(max(0, n - lookback_days)):
                    train = df2.iloc[:i+1][colname]
                    if train.shape[0] < 10:
                        continue
                    yhat = float(train.iloc[-1]) * (1 + float(train.pct_change().tail(7).mean()))
                    actual_idx = train.index[-1] + timedelta(days=1)
                    if actual_idx in df2.index:
                        preds.append(yhat); truths.append(float(df2.loc[actual_idx][colname]))
                if len(truths)==0: return None
                preds = np.array(preds); truths = np.array(truths)
                mae = mean_absolute_error(truths, preds)
                rmse = np.sqrt(mean_squared_error(truths, preds))
                mape = float(np.mean(np.abs((truths - preds) / truths)) * 100.0)
                return {"mae":mae, "rmse":rmse, "mape":mape, "n":len(truths)}

            st.subheader("Quick backtest")
            bt = backtest_simple(df_f, colname=cs.name, lookback_days=90)
            if bt:
                st.write(f"MAE: {bt['mae']:.4f}, RMSE: {bt['rmse']:.4f}, MAPE: {bt['mape']:.2f}% (n={bt['n']})")
            else:
                st.write("Backtest not available.")

# ---------- Portfolio ----------
with tab_portfolio:
    st.header("Portfolio")
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []

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
            st.session_state["portfolio"].append({
                "symbol": p_sym.strip().upper(),
                "qty": p_qty,
                "cost": p_cost,
                "added": datetime.now().isoformat()
            })
            st.success("Added.")

    st.subheader("Holdings")
    if len(st.session_state["portfolio"])==0:
        st.info("No holdings.")
    else:
        rows=[]; syms = list({p['symbol'] for p in st.session_state['portfolio']})
        price_map={}
        for s in syms:
            d, _ = fetch_data(s, period="7d", interval="1d")
            c = series_close(d).dropna()
            price_map[s] = float(c.iloc[-1]) * fx if not c.empty else None
        for p in st.session_state['portfolio']:
            cur = price_map.get(p['symbol'], None)
            val = cur * p['qty'] if cur is not None else None
            cost_total = p['cost'] * p['qty']
            pnl = (val - cost_total) if val is not None else None
            rows.append({
                "Symbol": p["symbol"],
                "Qty": p["qty"],
                "Cost/unit": fmt_currency(p["cost"], currency),
                "Current": fmt_currency(cur, currency) if cur is not None else "—",
                "Value": fmt_currency(val, currency) if val is not None else "—",
                "P&L": fmt_currency(pnl, currency) if pnl is not None else "—",
            })
        st.table(pd.DataFrame(rows))
        csv = pd.DataFrame(st.session_state["portfolio"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# ---------- News ----------
with tab_news:
    st.header("News")
    st.markdown("Paste an RSS feed or click fetch for CoinDesk headlines.")
    rss = st.text_input("RSS (optional):","")
    if st.button("Fetch"):
        url = rss.strip() if rss.strip() else "https://www.coindesk.com/arc/outboundfeeds/rss/"
        try:
            r = requests.get(url, timeout=8)
            if r.status_code==200:
                txt = r.text
                parts = txt.split("<title>")
                titles=[]
                for p in parts[1:10]:
                    t = p.split("</title>")[0]
                    if "CoinDesk" in t and len(titles) == 0:
                        continue
                    titles.append(t)
                st.write("Top headlines:")
                for t in titles[:8]:
                    st.write("-", t)
            else:
                st.error("Failed to fetch RSS.")
        except Exception as e:
            st.error("News fetch failed: " + str(e))

# ---------- About ----------
with tab_about:
    st.header("About")
    st.markdown("""
**Global Crypto Insight** — polished crypto dashboard.

- Dual-source data (Yahoo → CoinGecko fallback)  
- Live market, Forecasts (Prophet optional), Signals (MA/RSI/MACD)  
- Trend heatmap, Fear & Greed gauge, simple Portfolio & CSV export 1

Run: `streamlit run app.py`  
    """)
    st.caption("Educational only — not financial advice.")

# Footer
st.markdown("---")
st.caption("Manage your own risk. For commercial release prepare README & LICENSE.")
