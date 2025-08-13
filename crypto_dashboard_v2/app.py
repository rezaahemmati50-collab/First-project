# app.py
# Global Crypto Insight — Full, robust (gold + black theme)
# Single-file Streamlit app. Prophet & ta are OPTIONAL.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Global Crypto Insight", page_icon="🟡", layout="wide")

# ---------------- Optional libs ----------------
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

# ---------------- Utilities ----------------
def ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # yfinance گاهی Adj Close دارد ولی Close نه
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df = df.rename(columns={"Adj Close": "Close"})
    return df

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = ensure_flat_columns(df)
    df = df.reset_index()
    # نام ستون تاریخ را یکدست می‌کنیم
    if "Date" not in df.columns:
        # اولین ستون را Date فرض می‌کنیم
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Date"})
    # به datetime تبدیل کنیم
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # ستون‌های ضروری را اگر نیست، بسازیم تا کرش نکند
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan
    # فقط ردیف‌های با تاریخ معتبر
    df = df[df["Date"].notna()].sort_values("Date")
    return df

@st.cache_data(ttl=180, show_spinner=False)
def fetch_yf(symbol: str, period="3mo", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        return clean_ohlcv(df)
    except Exception:
        return pd.DataFrame()

def safe_last_two(series: pd.Series):
    """Return (last, prev) as floats or (np.nan, np.nan) if not enough data."""
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.shape[0] == 0:
            return (np.nan, np.nan)
        if s.shape[0] == 1:
            return (float(s.iloc[-1]), float(s.iloc[-1]))
        return (float(s.iloc[-1]), float(s.iloc[-2]))
    except Exception:
        return (np.nan, np.nan)

def moving_avg_forecast(series: pd.Series, days: int):
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return np.array([np.nan]*days)
        last = float(s.iloc[-1])
        pct_mean = s.pct_change().dropna().tail(30).mean()
        pct_mean = 0.0 if np.isnan(pct_mean) else float(pct_mean)
        return np.array([ last * ((1 + pct_mean) ** i) for i in range(1, days+1) ])
    except Exception:
        return np.array([np.nan]*days)

def simple_rsi(series, window=14):
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))

def simple_macd(series, fast=12, slow=26, signal=9):
    s = pd.to_numeric(series, errors="coerce")
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff

def compute_combined_signal(close_series: pd.Series, next_forecast=None):
    s = pd.to_numeric(close_series, errors="coerce").dropna()
    if s.empty:
        return ("NO DATA", "#9e9e9e", "insufficient data")

    ma20 = s.rolling(20, min_periods=1).mean().iloc[-1]
    ma50 = s.rolling(50, min_periods=1).mean().iloc[-1]
    ma200 = s.rolling(200, min_periods=1).mean().iloc[-1]

    # RSI / MACD (built-in یا ساده)
    try:
        if HAS_TA:
            rsi = ta.momentum.RSIIndicator(s, window=14).rsi().iloc[-1]
            macd_diff = ta.trend.MACD(s).macd_diff().iloc[-1]
        else:
            rsi = simple_rsi(s).iloc[-1]
            _, _, macd_diff_series = simple_macd(s)
            macd_diff = macd_diff_series.iloc[-1]
    except Exception:
        rsi, macd_diff = np.nan, np.nan

    score, reasons = 0, []

    # MAs
    try:
        if ma20 > ma50:
            score += 2; reasons.append("MA20>MA50")
        else:
            score -= 1; reasons.append("MA20<=MA50")
        if not np.isnan(ma200):
            if ma50 > ma200:
                score += 1; reasons.append("MA50>MA200")
            else:
                score -= 1; reasons.append("MA50<=MA200")
    except Exception:
        pass

    # RSI
    try:
        if not np.isnan(rsi):
            if rsi < 30:
                score += 1; reasons.append(f"RSI low ({rsi:.1f})")
            elif rsi > 70:
                score -= 1; reasons.append(f"RSI high ({rsi:.1f})")
    except Exception:
        pass

    # MACD
    try:
        if not np.isnan(macd_diff):
            if macd_diff > 0:
                score += 1; reasons.append("MACD positive")
            else:
                score -= 1; reasons.append("MACD negative")
    except Exception:
        pass

    # Forecast drift
    try:
        if next_forecast is not None and not np.isnan(next_forecast):
            last = float(s.iloc[-1])
            if last != 0:
                pct = (next_forecast - last) / last
                if pct > 0.01:
                    score += 1; reasons.append(f"Forecast +{pct*100:.2f}%")
                elif pct < -0.01:
                    score -= 1; reasons.append(f"Forecast {pct*100:.2f}%")
    except Exception:
        pass

    if score >= 4: return ("STRONG BUY", "#d4ffb3", " · ".join(reasons))
    if score >= 2: return ("BUY", "#b2ff66", " · ".join(reasons))
    if score == 1: return ("MILD BUY", "#ffe36b", " · ".join(reasons))
    if score == 0: return ("HOLD", "#cfd8dc", " · ".join(reasons))
    if score == -1: return ("MILD SELL", "#ffb86b", " · ".join(reasons))
    return ("SELL", "#ff7b7b", " · ".join(reasons))

@st.cache_data(ttl=600, show_spinner=False)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        if r.status_code == 200:
            j = r.json()
            if isinstance(j, dict) and "data" in j and len(j["data"]) > 0:
                e = j["data"][0]
                return {
                    "value": int(e.get("value", 50)),
                    "class": e.get("value_classification", "Neutral"),
                    "date": datetime.utcfromtimestamp(int(e.get("timestamp", 0))).strftime("%Y-%m-%d")
                            if e.get("timestamp") else None
                }
    except Exception:
        pass
    return {"value": None, "class": "N/A", "date": None}

def fmt_currency(x, cur="USD"):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{x:,.2f} {cur}"
    except Exception:
        return "—"

# ---------------- Header (gold + black) ----------------
st.markdown(
    """
    <style>
    .gci-header{
      padding:20px;border-radius:12px;
      background: linear-gradient(90deg,#0b0b0b 0%, #1a1a1a 50%, #3a2b10 100%);
      color: #f9f4ef; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .gci-title{font-size:26px;font-weight:800;color:#f5d76e}
    .gci-sub{color:#e6dccd;margin-top:6px}
    .kpi-card { background:linear-gradient(180deg,#2b2b2b,#171717); padding:12px;border-radius:8px; color:#fff; }
    .muted { color:#b6b6b6; }
    </style>
    <div class="gci-header">
      <div class="gci-title">Global Crypto Insight</div>
      <div class="gci-sub">Live market · Multi-model forecasts · Signals · Heatmap · Portfolio</div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
currency = st.sidebar.selectbox("Display currency", ["USD", "CAD", "EUR", "GBP"], index=0)
period = st.sidebar.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d", "1h"], index=0)

st.sidebar.markdown("---")
symbols_default = st.sidebar.text_area(
    "Symbols (comma separated)",
    value="BTC-USD,ETH-USD,ADA-USD,SOL-USD,XRP-USD"
).upper()
symbols = [s.strip() for s in symbols_default.split(",") if s.strip()]

st.sidebar.markdown("---")
manual = st.sidebar.text_input("Add single symbol (e.g. ADA-USD):", value="")
if manual:
    m = manual.strip().upper()
    if m not in symbols:
        symbols.insert(0, m)

st.sidebar.markdown("---")
st.sidebar.caption("Prophet & ta are optional. Install them for advanced forecasting and indicators.")

# ---------------- FX (USD -> target) ----------------
@st.cache_data(ttl=300, show_spinner=False)
def get_fx_usd_to(target: str) -> float:
    if target == "USD":
        return 1.0
    tick = {"CAD": "USDCAD=X", "EUR": "USDEUR=X", "GBP": "USDGBP=X"}.get(target)
    if not tick:
        return 1.0
    try:
        df = yf.download(tick, period="7d", interval="1d", progress=False)
        if df is None or df.empty:
            return 1.0
        df = ensure_flat_columns(df)
        close = pd.to_numeric(df.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
        return float(close.iloc[-1]) if not close.empty else 1.0
    except Exception:
        return 1.0

fx = get_fx_usd_to(currency)

# ---------------- Tabs ----------------
tab_market, tab_forecast, tab_portfolio, tab_news, tab_about = st.tabs(
    ["Market", "Forecast", "Portfolio", "News", "About"]
)

# ---------------- Market Tab ----------------
with tab_market:
    st.header("Market Overview")

    c1, c2, c3, c4 = st.columns([1.3, 2, 2, 2])
    # Fear & Greed
    with c1:
        fg = fetch_fear_greed()
        if fg["value"] is not None:
            figg = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=fg["value"],
                    title={"text": f"Fear & Greed ({fg['date']})"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#f5d76e"},
                        "steps": [
                            {"range": [0, 25], "color": "#5c1f1f"},
                            {"range": [25, 40], "color": "#9b5f00"},
                            {"range": [40, 60], "color": "#b0894a"},
                            {"range": [60, 75], "color": "#7bb383"},
                            {"range": [75, 100], "color": "#2b9348"},
                        ],
                    },
                )
            )
            figg.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figg, use_container_width=True)
        else:
            st.info("Fear & Greed: N/A")

    # Snapshot (first symbol)
    with c2:
        st.markdown("### Snapshot")
        if len(symbols) > 0:
            s0 = symbols[0]
            d0 = fetch_yf(s0, period=period, interval=interval)
            if not d0.empty and "Close" in d0.columns:
                last, prev = safe_last_two(d0["Close"])
                if not np.isnan(last) and not np.isnan(prev) and prev != 0:
                    st.metric(f"{s0} Latest ({currency})", f"{(last*fx):,.2f}", delta=f"{((last-prev)/prev*100):+.2f}%")
                else:
                    st.info(f"{s0}: insufficient data.")
            else:
                st.info(f"{s0}: no data.")
        else:
            st.info("No symbols configured.")

    # Period High
    with c3:
        st.markdown("### Period High")
        rows = []
        for s in symbols[:6]:
            d = fetch_yf(s, period=period, interval=interval)
            if not d.empty and "High" in d.columns:
                high = pd.to_numeric(d["High"], errors="coerce").dropna()
                if not high.empty:
                    rows.append(f"{s}: {(float(high.max())*fx):,.2f} {currency}")
        if rows:
            st.markdown("<br/>".join(rows), unsafe_allow_html=True)
        else:
            st.info("No high data.")

    # Period Low
    with c4:
        st.markdown("### Period Low")
        rows = []
        for s in symbols[:6]:
            d = fetch_yf(s, period=period, interval=interval)
            if not d.empty and "Low" in d.columns:
                low = pd.to_numeric(d["Low"], errors="coerce").dropna()
                if not low.empty:
                    rows.append(f"{s}: {(float(low.min())*fx):,.2f} {currency}")
        if rows:
            st.markdown("<br/>".join(rows), unsafe_allow_html=True)
        else:
            st.info("No low data.")

    st.markdown("---")

    # Summary table with signals
    st.subheader("Summary")
    summary = []
    with st.spinner("Fetching market data..."):
        for s in symbols:
            d = fetch_yf(s, period=period, interval=interval)
            if d.empty or "Close" not in d.columns:
                summary.append({"Symbol": s, "Price": None, "Change24h": None, "Signal": "NO DATA", "Color": "#9e9e9e"})
                continue
            d = d.sort_values("Date").reset_index(drop=True)
            last, prev = safe_last_two(d["Close"])
            price = last * fx if not np.isnan(last) else None
            change24 = ((last - prev) / prev * 100) if (not np.isnan(last) and not np.isnan(prev) and prev != 0) else 0.0
            fc = moving_avg_forecast(d["Close"], 1)
            next_fc = float(fc[0]) if len(fc) > 0 and not np.isnan(fc[0]) else None
            label, color, reason = compute_combined_signal(d["Close"], next_fc)
            summary.append({
                "Symbol": s,
                "Price": price,
                "Change24h": round(change24, 2) if not np.isnan(change24) else 0.0,
                "Signal": label,
                "Color": color,
                "Reason": reason
            })

    df_sum = pd.DataFrame(summary)
    if not df_sum.empty:
        df_sum["PriceStr"] = df_sum["Price"].apply(lambda v: fmt_currency(v, currency))
        df_sum["ChangeStr"] = df_sum["Change24h"].apply(lambda v: "—" if v is None or (isinstance(v,float) and np.isnan(v)) else f"{float(v):+.2f}%")
        st.dataframe(
            df_sum[["Symbol", "PriceStr", "ChangeStr", "Signal"]].rename(columns={
                "Symbol": "Symbol", "PriceStr": "Price", "ChangeStr": "24h", "Signal": "Signal"
            }),
            use_container_width=True
        )
    else:
        st.info("No data.")

    st.markdown("### Signal Cards")
    if not df_sum.empty:
        cols = st.columns(min(6, max(1, len(df_sum))))
        for i, row in df_sum.iterrows():
            c = cols[i % len(cols)]
            with c:
                html = f"""
                <div style='background:{row["Color"]};padding:10px;border-radius:8px;text-align:center;color:#021014;'>
                    <strong>{row["Symbol"]}</strong><br/>
                    {row["Signal"]}<br/>
                    {row["PriceStr"]} · {row["ChangeStr"]}
                </div>
                """
                st.markdown(html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Trend Heatmap (daily %)")
    heat_days = st.slider("Heatmap lookback (days)", min_value=5, max_value=30, value=14)
    heat_data = {}
    dates_idx = None
    for s in symbols:
        d = fetch_yf(s, period="2mo", interval="1d")
        if d.empty or "Date" not in d.columns or "Close" not in d.columns:
            continue
        ser = pd.to_numeric(d.set_index("Date")["Close"], errors="coerce").dropna()
        ser = ser.resample("D").ffill().dropna().tail(heat_days + 1)
        if ser.shape[0] < 2:
            continue
        rets = ser.pct_change().dropna() * 100
        heat_data[s] = rets
        dates_idx = rets.index if dates_idx is None else dates_idx.intersection(rets.index)

    if heat_data and dates_idx is not None and len(dates_idx) > 0:
        heat_df = pd.DataFrame(heat_data).loc[dates_idx].T
        if heat_df.shape[1] > 0:
            lastcol = heat_df.columns[-1]
            heat_df = heat_df.reindex(heat_df[lastcol].sort_values(ascending=False).index)
        fig_h = go.Figure(
            data=go.Heatmap(
                z=np.round(heat_df.values, 2),
                x=[d.strftime("%Y-%m-%d") for d in heat_df.columns],
                y=heat_df.index,
                colorscale="RdYlGn"
            )
        )
        fig_h.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Not enough data for heatmap.")

# ---------------- Forecast Tab ----------------
with tab_forecast:
    st.header("Forecast & Backtest")
    if len(symbols) == 0:
        st.info("Add symbols in the sidebar.")
    else:
        f_sym = st.selectbox("Choose symbol", options=symbols, index=0)
        f_period = st.selectbox("History period", ["3mo", "6mo", "1y", "2y"], index=0)
        f_interval = st.selectbox("Interval", ["1d", "1h"], index=0)
        f_model = st.selectbox("Model", ["Prophet (if installed)", "MovingAvg (fallback)"], index=0)
        f_horizon = st.selectbox("Horizon (days)", [3, 7, 14, 30], index=1)

        df_f = fetch_yf(f_sym, period=f_period, interval=f_interval)
        if df_f.empty or "Close" not in df_f.columns:
            st.warning("No historical data.")
        else:
            st.subheader(f"Historical: {f_sym}")
            st.line_chart(df_f.set_index("Date")["Close"])

            # Forecast
            with st.spinner("Running forecast..."):
                try:
                    if f_model.startswith("Prophet") and HAS_PROPHET and df_f.shape[0] >= 15:
                        pf = df_f[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
                        m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
                        m.fit(pf)
                        future = m.make_future_dataframe(periods=f_horizon, freq="D")
                        pred = m.predict(future)
                        tail = pred.tail(f_horizon)
                        forecast_vals = tail["yhat"].values
                        fc_dates = tail["ds"].dt.date.values
                    else:
                        forecast_vals = moving_avg_forecast(df_f["Close"], f_horizon)
                        last_date = df_f["Date"].iloc[-1]
                        fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]
                except Exception as e:
                    # fallback
                    forecast_vals = moving_avg_forecast(df_f["Close"], f_horizon)
                    last_date = df_f["Date"].iloc[-1]
                    fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]

            fc_table = pd.DataFrame({"Date": fc_dates, "Predicted": np.round(forecast_vals, 4)})
            st.subheader("Forecast")
            st.table(fc_table)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_f["Date"], y=df_f["Close"], mode="lines", name="History"))
            fig.add_trace(go.Scatter(x=fc_dates, y=forecast_vals, mode="lines+markers",
                                     name=f"Forecast {f_horizon}d", line=dict(dash="dash", color="#f5d76e")))
            fig.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(fig, use_container_width=True)

            # Backtest (quick)
            from sklearn.metrics import mean_absolute_error, mean_squared_error

            def backtest_simple(df_hist: pd.DataFrame, lookback_days=90):
                x = df_hist[["Date", "Close"]].copy()
                x["Close"] = pd.to_numeric(x["Close"], errors="coerce")
                x = x.dropna()
                x = x.set_index("Date").asfreq("D", method="pad").dropna()
                n = len(x)
                preds, truths = [], []
                for i in range(15, n-1):  # نیاز به حداقل تاریخچه
                    train = x.iloc[:i]["Close"]
                    yhat = train.iloc[-1] * (1 + train.pct_change().tail(7).mean())
                    target_idx = x.index[i] + timedelta(days=1)
                    if target_idx in x.index:
                        preds.append(float(yhat))
                        truths.append(float(x.loc[target_idx, "Close"]))
                    if len(preds) >= lookback_days:
                        break
                if len(truths) == 0:
                    return None
                preds = np.array(preds); truths = np.array(trruths if False else truths)  # guard
                mae = mean_absolute_error(truths, preds)
                rmse = np.sqrt(mean_squared_error(truths, preds))
                mape = float(np.mean(np.abs((truths - preds) / truths)) * 100)
                return {"mae": mae, "rmse": rmse, "mape": mape, "n": len(truths)}

            st.subheader("Quick backtest")
            bt = backtest_simple(df_f, lookback_days=80)
            if bt:
                st.write(f"MAE: {bt['mae']:.4f}, RMSE: {bt['rmse']:.4f}, MAPE: {bt['mape']:.2f}% (n={bt['n']})")
            else:
                st.write("Backtest not available.")

# ---------------- Portfolio ----------------
with tab_portfolio:
    st.header("Portfolio")
    if "portfolio" not in st.session_state:
        st.session_state["portfolio"] = []

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        p_sym = st.text_input("Symbol (e.g. ADA-USD):", "")
    with col2:
        p_qty = st.number_input("Quantity", min_value=0.0, value=0.0, step=0.01)
    with col3:
        p_cost = st.number_input(f"Cost per unit ({currency})", min_value=0.0, value=0.0, step=0.01)

    if st.button("Add to portfolio"):
        if p_sym.strip() == "" or p_qty <= 0 or p_cost <= 0:
            st.warning("Provide valid symbol, qty and cost.")
        else:
            st.session_state["portfolio"].append(
                {"symbol": p_sym.strip().upper(), "qty": p_qty, "cost": p_cost, "added": datetime.now().isoformat()}
            )
            st.success("Added.")

    st.subheader("Holdings")
    if len(st.session_state["portfolio"]) == 0:
        st.info("No holdings.")
    else:
        rows = []
        syms = list({p["symbol"] for p in st.session_state["portfolio"]})
        price_map = {}
        for s in syms:
            d = fetch_yf(s, period="7d", interval="1d")
            if not d.empty and "Close" in d.columns:
                last, _ = safe_last_two(d["Close"])
                price_map[s] = last * fx if not np.isnan(last) else None
            else:
                price_map[s] = None

        for p in st.session_state["portfolio"]:
            cur = price_map.get(p["symbol"], None)
            val = cur * p["qty"] if cur is not None else None
            cost_total = p["cost"] * p["qty"]
            pnl = (val - cost_total) if val is not None else None
            rows.append({
                "Symbol": p["symbol"],
                "Qty": p["qty"],
                "Cost/unit": fmt_currency(p["cost"], currency),
                "Current": fmt_currency(cur, currency),
                "Value": fmt_currency(val, currency),
                "P&L": fmt_currency(pnl, currency)
            })
        st.table(pd.DataFrame(rows))

        csv = pd.DataFrame(st.session_state["portfolio"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv,
                           file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

# ---------------- News ----------------
with tab_news:
    st.header("News")
    st.markdown("Paste an RSS feed or click fetch for CoinDesk headlines.")
    rss = st.text_input("RSS (optional):", "")
    if st.button("Fetch"):
        url = rss.strip() if rss.strip() else "https://www.coindesk.com/arc/outboundfeeds/rss/"
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                txt = r.text
                # Very simple parsing
                titles = []
                for chunk in txt.split("<item>")[1:10]:
                    if "<title>" in chunk and "</title>" in chunk:
                        t = chunk.split("<title>")[1].split("</title>")[0]
                        if "CDATA" in t:
                            t = t.replace("<![CDATA[", "").replace("]]>", "")
                        titles.append(t.strip())
                if not titles:
                    # fallback
                    parts = txt.split("<title>")
                    for p in parts[2:10]:
                        t = p.split("</title>")[0].strip()
                        if t:
                            titles.append(t)
                if titles:
                    st.write("Top headlines:")
                    for t in titles[:8]:
                        st.write("-", t)
                else:
                    st.info("No headlines found.")
            else:
                st.error("Failed to fetch RSS.")
        except Exception as e:
            st.error("News fetch failed.")

# ---------------- About ----------------
with tab_about:
    st.header("About")
    st.markdown("""
**Global Crypto Insight** — polished crypto dashboard.
- Live market, Forecasts (Prophet optional), Signals (MA/RSI/MACD)
- Trend heatmap, Fear & Greed gauge, simple Portfolio & CSV export
""")
    st.markdown("Run locally: `streamlit run app.py`")
    st.caption("Educational only — not financial advice.")

# ---------------- Footer ----------------
st.markdown("---")
st.caption("Manage your own risk. For commercial release prepare README & LICENSE.")
