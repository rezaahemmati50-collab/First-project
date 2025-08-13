# app.py
# Global Crypto Insight — Top50 + Market page (all-in-one)
# Single-file Streamlit app. Prophet & ta optional.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Global Crypto Insight", page_icon="🟡", layout="wide")

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

# ---------- Helpers ----------
def ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance multi-ticker comes as ('Close','BTC-USD') etc.
        df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    return df

@st.cache_data(ttl=180)
def fetch_yf(symbol: str, period="3mo", interval="1d") -> pd.DataFrame:
    """Download one ticker safely."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = ensure_flat_columns(df)
        df = df.reset_index()
        first_col = df.columns[0]
        if str(first_col).lower() != "date":
            df.rename(columns={first_col: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()

def moving_avg_forecast(series: pd.Series, days: int):
    try:
        s = series.dropna()
        if s.empty:
            return np.array([np.nan] * days)
        last = float(s.iloc[-1])
        pct = s.pct_change().dropna()
        avg_pct = float(pct.mean()) if not pct.empty else 0.0
        return np.array([last * ((1 + avg_pct) ** i) for i in range(1, days + 1)])
    except Exception:
        return np.array([np.nan] * days)

def simple_rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))

def simple_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff

def compute_combined_signal(close_series, next_forecast=None):
    if close_series.dropna().empty:
        return ("NO DATA", "#9e9e9e", "insufficient data")
    cs = close_series.dropna()
    ma20 = cs.rolling(20, min_periods=1).mean().iloc[-1]
    ma50 = cs.rolling(50, min_periods=1).mean().iloc[-1]
    ma200 = cs.rolling(200, min_periods=1).mean().iloc[-1]
    try:
        if HAS_TA:
            rsi = ta.momentum.RSIIndicator(cs, window=14).rsi().iloc[-1]
            macd_diff = ta.trend.MACD(cs).macd_diff().iloc[-1]
        else:
            rsi = simple_rsi(cs).iloc[-1]
            _, _, macd_diff_series = simple_macd(cs)
            macd_diff = macd_diff_series.iloc[-1]
    except Exception:
        rsi = np.nan
        macd_diff = np.nan

    score = 0
    reasons = []
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
    try:
        if rsi < 30:
            score += 1; reasons.append(f"RSI low ({rsi:.1f})")
        elif rsi > 70:
            score -= 1; reasons.append(f"RSI high ({rsi:.1f})")
    except Exception:
        pass
    try:
        if macd_diff > 0:
            score += 1; reasons.append("MACD positive")
        else:
            score -= 1; reasons.append("MACD negative")
    except Exception:
        pass
    try:
        if next_forecast is not None and not np.isnan(next_forecast):
            last = float(cs.iloc[-1])
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

@st.cache_data(ttl=600)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        if r.status_code == 200:
            j = r.json()
            if "data" in j and len(j["data"]) > 0:
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
        return f"{x:,.2f} {cur}"
    except Exception:
        return "—"

# -------- Top coins (CoinGecko) --------
@st.cache_data(ttl=900)
def get_top_coins(limit=50):
    url = f"https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc",
              "per_page": int(limit), "page": 1, "sparkline": "false"}
    out = []
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for item in data:
                sym = str(item.get("symbol","")).upper()
                name = item.get("name","")
                ysym = f"{sym}-USD"  # Yahoo symbol guess
                out.append({"symbol": ysym, "name": name, "cg_symbol": sym})
    except Exception:
        pass
    # تضمین: همیشه چند مورد داشته باشیم
    if not out:
        out = [{"symbol": s, "name": s.replace("-USD","")} for s in
               ["BTC-USD","ETH-USD","ADA-USD","BNB-USD","SOL-USD","XRP-USD","DOGE-USD","TRX-USD","TON-USD","LINK-USD"]]
    return out

# ---------- Sidebar ----------
st.sidebar.header("Settings")
use_top50 = st.sidebar.checkbox("Use Top 50 from CoinGecko", value=True)
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)

st.sidebar.markdown("---")
if use_top50:
    top = get_top_coins(50)
    symbols = [t["symbol"] for t in top]
else:
    symbols_default = st.sidebar.text_area("Symbols (comma separated)", value="BTC-USD,ETH-USD,ADA-USD,SOL-USD,XRP-USD").upper()
    symbols = [s.strip() for s in symbols_default.split(",") if s.strip()]

st.sidebar.caption("Prophet & ta are optional. Install them for advanced forecasting and indicators.")

# FX (display currency)
@st.cache_data(ttl=300)
def get_fx(target):
    if target == "USD": return 1.0
    mp = {"CAD":"USDCAD=X","EUR":"USDEUR=X","GBP":"USDGBP=X"}
    t = mp.get(target)
    if not t: return 1.0
    try:
        df = yf.download(t, period="5d", interval="1d", progress=False)
        if df is None or df.empty: return 1.0
        return float(df["Close"].dropna().iloc[-1])
    except Exception:
        return 1.0

currency = st.sidebar.selectbox("Display currency", ["USD","CAD","EUR","GBP"], index=0)
fx = get_fx(currency)

# ---------- Header ----------
st.markdown(
    """
    <style>
    .gci-header{
      padding:16px;border-radius:12px;
      background: linear-gradient(90deg,#0b0b0b 0%, #1a1a1a 50%, #3a2b10 100%);
      color: #f9f4ef; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .gci-title{font-size:26px;font-weight:800;color:#f5d76e}
    .gci-sub{color:#e6dccd;margin-top:6px}
    </style>
    <div class="gci-header">
      <div class="gci-title">Global Crypto Insight</div>
      <div class="gci-sub">Top 50 · Signals · Forecast · Heatmap · Portfolio</div>
    </div>
    """, unsafe_allow_html=True
)

# ---------- Tabs ----------
tabs = st.tabs(["Market","Forecast","Portfolio","News","About"])
tab_market, tab_forecast, tab_portfolio, tab_news, tab_about = tabs

# ================= Market Tab =================
with tab_market:
    st.header("Market Overview (Top 50)")
    # Fear & Greed
    fg = fetch_fear_greed()
    c1,c2 = st.columns([1,3])
    with c1:
        if fg["value"] is not None:
            figg = go.Figure(go.Indicator(
                mode="gauge+number", value=fg["value"],
                title={"text":f"Fear & Greed ({fg['date']})"},
                gauge={"axis":{"range":[0,100]}, "bar":{"color":"#f5d76e"},
                       "steps":[{"range":[0,25],"color":"#5c1f1f"},
                                {"range":[25,40],"color":"#9b5f00"},
                                {"range":[40,60],"color":"#b0894a"},
                                {"range":[60,75],"color":"#7bb383"},
                                {"range":[75,100],"color":"#2b9348"}]}))
            figg.update_layout(height=220, margin=dict(l=10,r=10,t=10,b=10), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figg, use_container_width=True)
        else:
            st.info("Fear & Greed: N/A")
    with c2:
        st.caption("قیمت‌ها از یاهوفایننس، لیست رتبه‌بندی از CoinGecko. اگر داده نمیدهد، نماد در یاهو موجود نیست و از جدول حذف می‌شود.")

    st.markdown("---")
    # Summary table for all symbols
    rows = []
    with st.spinner("Fetching prices & computing signals..."):
        for s in symbols:
            d = fetch_yf(s, period=period, interval=interval)
            if d.empty or "Close" not in d.columns:
                continue
            d = d.sort_values("Date")
            close = d["Close"].dropna()
            if close.shape[0] == 0:
                continue
            price_usd = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if close.shape[0] >= 2 else price_usd
            ch24 = (price_usd - prev) / prev * 100 if prev != 0 else 0.0
            # mini-forecast next step (moving avg)
            fc = moving_avg_forecast(close, 1)
            next_fc = float(fc[0]) if len(fc) else None
            label, color, reason = compute_combined_signal(close, next_fc)
            rows.append({
                "Symbol": s,
                "Price": price_usd * fx,
                "Change24h": round(ch24, 2),
                "Signal": label,
                "Color": color,
                "Reason": reason
            })
    if len(rows) == 0:
        st.warning("No symbols with data.")
    else:
        df_sum = pd.DataFrame(rows)
        # sort by market move last 24h desc
        df_sum = df_sum.sort_values("Change24h", ascending=False).reset_index(drop=True)
        df_show = df_sum[["Symbol"]].copy()
        df_show["Price"] = df_sum["Price"].apply(lambda v: fmt_currency(v, currency))
        df_show["24h"] = df_sum["Change24h"].map(lambda v: f"{v:+.2f}%")
        df_show["Signal"] = df_sum["Signal"]
        st.dataframe(df_show, use_container_width=True, height=550)

        st.markdown("### Signal Cards")
        cols = st.columns(6)
        for i, row in df_sum.iterrows():
            c = cols[i % 6]
            with c:
                st.markdown(
                    f"<div style='background:{row['Color']};padding:10px;border-radius:8px;text-align:center;color:#021014;'>"
                    f"<strong>{row['Symbol']}</strong><br/>{row['Signal']}<br/>{fmt_currency(row['Price'],currency)} · {row['Change24h']:+.2f}%"
                    f"</div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.subheader("Trend Heatmap (daily %)")
    heat_days = st.slider("Heatmap lookback (days)", min_value=5, max_value=30, value=14)
    heat_data = {}
    dates_idx = None
    for s in symbols[:50]:
        d = fetch_yf(s, period="2mo", interval="1d")
        if d.empty or "Date" not in d.columns or "Close" not in d.columns:
            continue
        ser = d.set_index("Date")["Close"].resample("D").ffill().dropna()
        ser = ser.tail(heat_days + 1)
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
        fig_h = go.Figure(data=go.Heatmap(
            z=np.round(heat_df.values, 2),
            x=[d.strftime("%Y-%m-%d") for d in heat_df.columns],
            y=heat_df.index, colorscale="RdYlGn"))
        fig_h.update_layout(height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig_h, use_container_width=True)
    else:
        st.info("Not enough data for heatmap.")

# ================= Forecast Tab =================
with tab_forecast:
    st.header("Forecast & Backtest")
    f_sym = st.selectbox("Choose symbol", options=symbols, index=0 if symbols else None)
    f_period = st.selectbox("History period", ["3mo","6mo","1y","2y"], index=0)
    f_interval = st.selectbox("Interval", ["1d","1h"], index=0)
    # Default = MovingAvg
    f_model = st.selectbox("Model", ["MovingAvg (fallback)", "Prophet (if installed)"], index=0)
    f_horizon = st.selectbox("Horizon (days)", [3,7,14,30], index=1)

    if f_sym:
        df_f = fetch_yf(f_sym, period=f_period, interval=f_interval)
        if df_f.empty or "Close" not in df_f.columns:
            st.warning("No historical data.")
        else:
            st.subheader(f"Historical: {f_sym}")
            st.line_chart(df_f.set_index("Date")["Close"])
            # forecast
            with st.spinner("Running forecast..."):
                try:
                    if f_model.startswith("Prophet") and HAS_PROPHET and df_f.shape[0] >= 10:
                        pf = df_f[["Date","Close"]].rename(columns={"Date":"ds","Close":"y"})
                        m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
                        m.fit(pf)
                        future = m.make_future_dataframe(periods=f_horizon, freq="D")
                        pred = m.predict(future)
                        tail = pred.tail(f_horizon)
                        forecast_vals = tail["yhat"].values
                        fc_dates = tail["ds"].dt.date.values
                    else:
                        arr = moving_avg_forecast(df_f["Close"], f_horizon)
                        forecast_vals = arr
                        last_date = df_f["Date"].iloc[-1]
                        fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]
                except Exception as e:
                    arr = moving_avg_forecast(df_f["Close"], f_horizon)
                    forecast_vals = arr
                    last_date = df_f["Date"].iloc[-1]
                    fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]

            fc_table = pd.DataFrame({"Date":fc_dates, "Predicted":np.round(forecast_vals,4)})
            st.subheader("Forecast")
            st.table(fc_table)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_f["Date"], y=df_f["Close"], mode="lines", name="History"))
            fig.add_trace(go.Scatter(x=fc_dates, y=forecast_vals, mode="lines+markers",
                                     name=f"Forecast {f_horizon}d", line=dict(dash="dash", color="#f5d76e")))
            fig.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(fig, use_container_width=True)

            # quick backtest
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            def backtest_simple(df_hist, horizon=1, lookback_days=60):
                df2 = df_hist[["Date","Close"]].set_index("Date").asfreq("D", method="pad").dropna()
                n = len(df2); preds=[]; truths=[]
                for i in range(max(0, n - lookback_days)):
                    train = df2.iloc[:i+1]["Close"]
                    if train.shape[0] < 6: 
                        continue
                    yhat = train.iloc[-1] * (1 + train.pct_change().tail(7).mean())
                    actual_idx = train.index[-1] + timedelta(days=1)
                    if actual_idx in df2.index:
                        preds.append(yhat); truths.append(df2.loc[actual_idx]["Close"])
                if len(truths) == 0:
                    return None
                preds = np.array(preds); truths = np.array(truths)
                mae = mean_absolute_error(truths, preds)
                rmse = np.sqrt(mean_squared_error(truths, preds))
                mape = np.mean(np.abs((truths - preds) / truths)) * 100
                return {"mae":mae, "rmse":rmse, "mape":mape, "n":len(truths)}
            st.subheader("Quick backtest")
            bt = backtest_simple(df_f, horizon=1, lookback_days=60)
            if bt:
                st.write(f"MAE: {bt['mae']:.4f}, RMSE: {bt['rmse']:.4f}, MAPE: {bt['mape']:.2f}% (n={bt['n']})")
            else:
                st.write("Backtest not available.")

# ================= Portfolio =================
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
            st.session_state["portfolio"].append({"symbol":p_sym.strip().upper(),"qty":p_qty,"cost":p_cost,"added":datetime.now().isoformat()})
            st.success("Added.")
    st.subheader("Holdings")
    if len(st.session_state["portfolio"])==0:
        st.info("No holdings.")
    else:
        rows=[]; syms = list({p["symbol"] for p in st.session_state["portfolio"]})
        price_map={}
        for s in syms:
            d = fetch_yf(s, period="7d", interval="1d")
            if not d.empty and "Close" in d.columns:
                price_map[s] = float(d["Close"].dropna().iloc[-1]) * fx
            else:
                price_map[s] = None
        for p in st.session_state["portfolio"]:
            cur = price_map.get(p["symbol"], None)
            val = cur * p["qty"] if cur is not None else None
            cost_total = p["cost"] * p["qty"]
            pnl = val - cost_total if val is not None else None
            rows.append({
                "Symbol":p["symbol"],
                "Qty":p["qty"],
                "Cost/unit":fmt_currency(p["cost"],currency),
                "Current":fmt_currency(cur,currency) if cur else "—",
                "Value":fmt_currency(val,currency) if val else "—",
                "P&L":fmt_currency(pnl,currency) if pnl else "—"
            })
        st.table(pd.DataFrame(rows))
        csv = pd.DataFrame(st.session_state["portfolio"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv,
                           file_name=f"portfolio_{datetime.now().strftime('%Y%m%d')}.csv",
                           mime="text/csv")

# ================= News =================
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
                for p in parts[1:8]:
                    t = p.split("</title>")[0]
                    titles.append(t)
                st.write("Top headlines:")
                for t in titles: st.write("-",t)
            else:
                st.error("Failed to fetch RSS.")
        except Exception as e:
            st.error("News fetch failed: " + str(e))

# ================= About =================
with tab_about:
    st.header("About")
    st.markdown("""
    **Global Crypto Insight** — polished crypto dashboard.
    - Live market (Top 50), Forecasts (Prophet optional), Signals (MA/RSI/MACD)
    - Trend heatmap, Fear & Greed gauge, simple Portfolio & CSV export
    """)
    st.markdown("Run: `streamlit run app.py`")
    st.caption("Educational only — not financial advice.")

# Footer
st.markdown("---")
st.caption("Manage your own risk. For commercial release prepare README & LICENSE.")
