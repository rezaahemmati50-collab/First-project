# app.py
# Global Crypto Insight — Top50 + improved MovingAvg forecast
# Dependencies:
# streamlit, yfinance, pandas, numpy, plotly, requests, scikit-learn
# Optional: prophet, ta

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from time import sleep

st.set_page_config(page_title="Global Crypto Insight", page_icon="🟡", layout="wide")

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
    # If multiindex columns from yf.download, normalize into (ticker, field) or field only
    if isinstance(df.columns, pd.MultiIndex):
        # produce dict-like per ticker later; leave as-is for parsing function
        return df
    return df

@st.cache_data(ttl=600)
def fetch_top_coins_coingecko(n=50):
    """Return top n coins from CoinGecko (ids and symbols)."""
    url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={n}&page=1&sparkline=false"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        # symbol returned like 'btc' -> convert to uppercase and append -USD
        symbols = []
        for item in data:
            sym = item.get("symbol", "").upper()
            if sym:
                symbols.append(f"{sym}-USD")
        return symbols
    except Exception:
        return []

def chunked(iterable, size):
    it = list(iterable)
    for i in range(0, len(it), size):
        yield it[i:i+size]

@st.cache_data(ttl=600)
def fetch_yf_batch(tickers, period="3mo", interval="1d", batch_size=10):
    """
    Fetch data for tickers in batches using yfinance.download with group_by='ticker'.
    Return dict: symbol -> DataFrame with Date column.
    """
    out = {}
    if not tickers:
        return out
    for batch in chunked(tickers, batch_size):
        try:
            # yf.download accepts list
            df = yf.download(batch, period=period, interval=interval, group_by='ticker', progress=False, threads=True)
            if df is None or df.empty:
                # if nothing returned for the batch, skip with small pause
                sleep(0.5)
                continue
            # handle two cases:
            # 1) group_by='ticker' => top level columns are tickers
            # 2) MultiIndex with fields first level ('Close','Open') and tickers second -> handle both
            if isinstance(df.columns, pd.MultiIndex):
                # check which layout
                top_levels = list(df.columns.levels[0])
                second_levels = list(df.columns.levels[1])
                # Two possibilities: (field, ticker) or (ticker, field)
                # Decide by checking if 'Close' in top_levels
                if 'Close' in top_levels:
                    # layout (field, ticker)
                    for field in top_levels:
                        if field != 'Close': continue
                        for ticker in df[field].columns:
                            sub = df[field][ticker].copy().reset_index().rename(columns={"index":"Date", field:"Close"})
                            sub['Date'] = pd.to_datetime(sub['Date'], errors='coerce')
                            out[ticker] = sub
                else:
                    # assume layout (ticker, field)
                    for ticker in df.columns.levels[0]:
                        try:
                            sub = df[ticker].copy().reset_index()
                            if 'Date' in sub.columns:
                                sub['Date'] = pd.to_datetime(sub['Date'], errors='coerce')
                            out[ticker] = sub
                        except Exception:
                            continue
            else:
                # single ticker in batch -> df has columns like Close, Open
                # iterate over batch
                for ticker in batch:
                    # try to slice by ticker name in columns (some yfinance returns columns with ticker suffix)
                    # If only one ticker requested, df belongs to it
                    try:
                        sub = df.copy().reset_index()
                        if 'Date' in sub.columns:
                            sub['Date'] = pd.to_datetime(sub['Date'], errors='coerce')
                        out[ticker] = sub
                    except Exception:
                        continue
        except Exception:
            # on error continue with next batch
            sleep(0.5)
            continue
        sleep(0.2)  # small throttle
    return out

def moving_avg_forecast_improved(series, days=3):
    """
    Improved moving-average-based forecast:
    - compute average pct changes for short, medium, long windows (if available)
    - combine them weighted to produce a smoothed forecast
    """
    try:
        s = series.dropna().astype(float)
        if s.empty:
            return np.array([np.nan]*days)
        last = float(s.iloc[-1])
        # pct changes series
        pct = s.pct_change().dropna()
        if pct.empty:
            return np.array([last]*days)
        # windows
        w_short = pct.tail(3).mean() if pct.shape[0] >= 1 else 0.0
        w_med = pct.tail(7).mean() if pct.shape[0] >= 3 else w_short
        w_long = pct.tail(30).mean() if pct.shape[0] >= 7 else w_med
        # weights favor short-term but include longer term to stabilize
        weights = np.array([0.5, 0.3, 0.2])
        combined = weights[0]*w_short + weights[1]*w_med + weights[2]*w_long
        # if combined is NaN fallback to mean
        if np.isnan(combined):
            combined = pct.mean()
        forecasts = []
        cur = last
        for i in range(days):
            cur = cur * (1 + combined)
            forecasts.append(cur)
        return np.array(forecasts)
    except Exception:
        return np.array([np.nan]*days)

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
        return ("NO DATA","#9e9e9e","insufficient data")
    cs = close_series.dropna().astype(float)
    ma20 = cs.rolling(20, min_periods=1).mean().iloc[-1]
    ma50 = cs.rolling(50, min_periods=1).mean().iloc[-1]
    ma200 = cs.rolling(200, min_periods=1).mean().iloc[-1]
    try:
        if HAS_TA:
            rsi = ta.momentum.RSIIndicator(cs, window=14).rsi().iloc[-1]
            macd_diff = ta.trend.MACD(cs).macd_diff().iloc[-1]
        else:
            rsi = simple_rsi(cs).iloc[-1]
            _,_,macd_diff_series = simple_macd(cs)
            macd_diff = macd_diff_series.iloc[-1]
    except Exception:
        rsi = np.nan; macd_diff = np.nan

    score = 0; reasons = []
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

    if score >= 4: return ("STRONG BUY","#d4ffb3"," · ".join(reasons))
    if score >= 2: return ("BUY","#b2ff66"," · ".join(reasons))
    if score == 1: return ("MILD BUY","#ffe36b"," · ".join(reasons))
    if score == 0: return ("HOLD","#cfd8dc"," · ".join(reasons))
    if score == -1: return ("MILD SELL","#ffb86b"," · ".join(reasons))
    return ("SELL","#ff7b7b"," · ".join(reasons))

@st.cache_data(ttl=300)
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

def fmt_currency(x, cur="USD"):
    try:
        return f"{x:,.2f} {cur}"
    except Exception:
        return "—"

# ---------------- Header ----------------
st.markdown(
    """
    <style>
    .gci-header{
      padding:18px;border-radius:10px;
      background: linear-gradient(90deg,#0b0b0b 0%, #151515 50%, #3a2b10 100%);
      color: #f9f4ef; box-shadow: 0 8px 20px rgba(0,0,0,0.4);
    }
    .gci-title{font-size:24px;font-weight:800;color:#f5d76e}
    .gci-sub{color:#e6dccd;margin-top:6px}
    </style>
    <div class="gci-header">
      <div class="gci-title">Global Crypto Insight</div>
      <div class="gci-sub">Top coins · Improved MovingAvg forecast · Signals · Heatmap</div>
    </div>
    """, unsafe_allow_html=True
)

# ---------------- Sidebar ----------------
st.sidebar.header("Settings")
currency = st.sidebar.selectbox("Display currency", ["USD","CAD","EUR","GBP"], index=0)
use_coingecko = st.sidebar.checkbox("Use top coins from CoinGecko", value=True)
top_n = st.sidebar.selectbox("Top N coins", [20,30,40,50], index=3 if use_coingecko else 0)
period = st.sidebar.selectbox("History period", ["1mo","3mo","6mo","1y","2y"], index=1)
interval = st.sidebar.selectbox("Interval", ["1d","1h"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption("Model: default MovingAvg (improved). Prophet optional (install prophet to enable).")

manual_symbols = st.sidebar.text_area("Extra symbols (comma separated, e.g. BTC-USD,ADA-USD)", value="")
manual_list = [s.strip().upper() for s in manual_symbols.split(",") if s.strip()]

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
tabs = st.tabs(["Market","Forecast","Portfolio","News","About"])
tab_market, tab_forecast, tab_portfolio, tab_news, tab_about = tabs

# ---------------- Market Tab ----------------
with tab_market:
    st.header("Market Overview")
    fg = fetch_fear_greed()
    c1,c2,c3,c4 = st.columns([1.2,2,2,2])
    with c1:
        if fg['value'] is not None:
            figg = go.Figure(go.Indicator(mode="gauge+number", value=fg['value'],
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
        st.markdown("### Snapshot")
        st.info("Select a symbol in Forecast tab to see detailed chart & forecast.")
    with c3:
        st.markdown("### Period High (sample)")
        st.write("Top symbols will be listed below after fetch.")
    with c4:
        st.markdown("### Period Low (sample)")
        st.write("—")

    st.markdown("---")
    st.subheader("Symbols & Signals")

    # build symbol list
    symbols = []
    if use_coingecko:
        cg = fetch_top_coins_coingecko(top_n)
        symbols.extend(cg)
    symbols = list(dict.fromkeys(symbols + manual_list))  # unique, preserve order

    if not symbols:
        st.warning("No symbols selected.")
    else:
        st.info(f"Fetching data for up to {len(symbols)} symbols (batch)...")
        # fetch in batches
        data_map = fetch_yf_batch(symbols, period=period, interval=interval, batch_size=8)

        # filter out empty / no Close
        good_symbols = []
        summary = []
        for s in symbols:
            df = data_map.get(s)
            if df is None or df.empty:
                continue
            # ensure Close column exists
            if 'Close' not in df.columns:
                # try common alternatives
                possible = [c for c in df.columns if 'Close' in c]
                if possible:
                    df = df.rename(columns={possible[0]:'Close'})
                else:
                    continue
            # dropna
            if df['Close'].dropna().empty:
                continue
            # accept
            good_symbols.append(s)
            # prepare stats
            df_sorted = df.sort_values('Date').reset_index(drop=True)
            price_usd = float(df_sorted['Close'].dropna().iloc[-1])
            prev = float(df_sorted['Close'].dropna().iloc[-2]) if df_sorted['Close'].dropna().shape[0]>=2 else price_usd
            change24 = (price_usd - prev)/prev*100 if prev!=0 else 0.0
            fc = moving_avg_forecast_improved(df_sorted['Close'], days=1)
            next_fc = float(fc[0]) if len(fc)>0 else np.nan
            label,color,reason = compute_combined_signal(df_sorted['Close'], next_fc)
            summary.append({"Symbol":s,"Price":price_usd*fx,"Change24h":round(change24,2),"Signal":label,"Color":color,"Reason":reason})

        if not good_symbols:
            st.error("No symbols with data (Yahoo didn't return Close prices). Try fewer symbols or change period/interval.")
        else:
            df_sum = pd.DataFrame(summary)
            df_sum['PriceStr'] = df_sum['Price'].apply(lambda v: fmt_currency(v,currency) if v is not None else "—")
            df_sum['ChangeStr'] = df_sum['Change24h'].apply(lambda v: f"{v:+.2f}%")
            st.dataframe(df_sum[['Symbol','PriceStr','ChangeStr','Signal']].rename(columns={"Symbol":"Symbol","PriceStr":"Price","ChangeStr":"24h","Signal":"Signal"}), use_container_width=True)

            st.markdown("### Signal Cards")
            cols = st.columns(min(6, max(1,len(df_sum))))
            for i,row in df_sum.iterrows():
                c = cols[i % len(cols)]
                with c:
                    html = f"<div style='background:{row['Color']};padding:10px;border-radius:8px;text-align:center;color:#021014;'><strong>{row['Symbol']}</strong><br/>{row['Signal']}<br/>{row['PriceStr']} · {row['ChangeStr']}</div>"
                    st.markdown(html, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Trend Heatmap (daily %)")
            heat_days = st.slider("Heatmap lookback (days)", min_value=5, max_value=30, value=14)
            heat_data = {}
            dates_idx = None
            for s in good_symbols:
                d = data_map.get(s)
                try:
                    ser = d.set_index('Date')['Close'].resample('D').ffill().dropna()
                    ser = ser.tail(heat_days+1)
                    if ser.shape[0] < 2:
                        continue
                    rets = ser.pct_change().dropna()*100
                    heat_data[s] = rets
                    dates_idx = rets.index if dates_idx is None else dates_idx.intersection(rets.index)
                except Exception:
                    continue
            if heat_data and dates_idx is not None and len(dates_idx)>0:
                heat_df = pd.DataFrame(heat_data).loc[dates_idx].T
                if heat_df.shape[1] > 0:
                    lastcol = heat_df.columns[-1]
                    heat_df = heat_df.reindex(heat_df[lastcol].sort_values(ascending=False).index)
                fig_h = go.Figure(data=go.Heatmap(z=np.round(heat_df.values,2), x=[d.strftime("%Y-%m-%d") for d in heat_df.columns], y=heat_df.index, colorscale='RdYlGn'))
                fig_h.update_layout(height=320, margin=dict(t=10,b=10))
                st.plotly_chart(fig_h, use_container_width=True)
            else:
                st.info("Not enough series data for heatmap.")

# ---------------- Forecast Tab ----------------
with tab_forecast:
    st.header("Forecast & Backtest")
    # select symbol from good ones (we'll re-fetch if needed)
    all_symbols = []
    if use_coingecko:
        all_symbols = fetch_top_coins_coingecko(top_n)
    all_symbols = list(dict.fromkeys(all_symbols + manual_list))
    if not all_symbols:
        st.info("No symbols configured.")
    f_sym = st.selectbox("Choose symbol", options=all_symbols, index=0 if all_symbols else None)
    f_period = st.selectbox("History period", ["3mo","6mo","1y","2y"], index=0)
    f_interval = st.selectbox("Interval", ["1d","1h"], index=0)
    f_model = st.selectbox("Model", ["MovingAvg (improved)","Prophet (if installed)"], index=0)
    f_horizon = st.selectbox("Horizon (days)", [3,7,14,30], index=1)

    if f_sym:
        df_f_map = fetch_yf_batch([f_sym], period=f_period, interval=f_interval, batch_size=1)
        df_f = df_f_map.get(f_sym)
        if df_f is None or df_f.empty or 'Close' not in df_f.columns or df_f['Close'].dropna().empty:
            st.warning("No historical data for selected symbol.")
        else:
            st.subheader(f"Historical: {f_sym}")
            st.line_chart(df_f.sort_values('Date').set_index('Date')['Close'])

            # forecast
            forecast_vals = None
            fc_dates = None
            with st.spinner("Running forecast..."):
                try:
                    if f_model.startswith("Prophet") and HAS_PROPHET and df_f.shape[0] >= 10:
                        pf = df_f[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
                        pf['y'] = pd.to_numeric(pf['y'], errors='coerce')
                        pf = pf.dropna()
                        if pf.shape[0] >= 10:
                            m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
                            m.fit(pf)
                            future = m.make_future_dataframe(periods=f_horizon, freq='D')
                            pred = m.predict(future)
                            tail = pred.tail(f_horizon)
                            forecast_vals = tail['yhat'].values
                            fc_dates = tail['ds'].dt.date.values
                        else:
                            # fallback
                            arr = moving_avg_forecast_improved(df_f['Close'], f_horizon)
                            forecast_vals = arr
                            last_date = df_f['Date'].iloc[-1]
                            fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]
                    else:
                        arr = moving_avg_forecast_improved(df_f['Close'], f_horizon)
                        forecast_vals = arr
                        last_date = df_f['Date'].iloc[-1]
                        fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]
                except Exception as e:
                    st.error("Forecast error (fallback): " + str(e))
                    arr = moving_avg_forecast_improved(df_f['Close'], f_horizon)
                    forecast_vals = arr
                    last_date = df_f['Date'].iloc[-1]
                    fc_dates = [(last_date + timedelta(days=i+1)).date() for i in range(f_horizon)]

            fc_table = pd.DataFrame({"Date":fc_dates, "Predicted":np.round(forecast_vals,6)})
            st.subheader("Forecast")
            st.table(fc_table)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_f['Date'], y=df_f['Close'], mode='lines', name='History'))
            fig.add_trace(go.Scatter(x=fc_dates, y=forecast_vals, mode='lines+markers', name=f'Forecast {f_horizon}d', line=dict(dash='dash', color='#f5d76e')))
            fig.update_layout(template='plotly_dark', height=520)
            st.plotly_chart(fig, use_container_width=True)

            # backtest quick
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            def backtest_simple(df_hist, horizon=1, lookback_days=60):
                df2 = df_hist[['Date','Close']].set_index('Date').asfreq('D', method='pad').dropna()
                n = len(df2); preds=[]; truths=[]
                for i in range(max(0,n - lookback_days)):
                    train = df2.iloc[:i+1]['Close']
                    if train.shape[0] < 6: continue
                    yhat = train.iloc[-1] * (1 + train.pct_change().tail(7).mean())
                    actual_idx = train.index[-1] + timedelta(days=1)
                    if actual_idx in df2.index:
                        preds.append(yhat); truths.append(df2.loc[actual_idx]['Close'])
                if len(truths)==0: return None
                preds = np.array(preds); truths = np.array(truths)
                mae = mean_absolute_error(truths, preds)
                rmse = np.sqrt(mean_squared_error(truths, preds))
                mape = np.mean(np.abs((truths - preds) / truths)) * 100
                return {"mae":mae,"rmse":rmse,"mape":mape,"n":len(truths)}
            st.subheader("Quick backtest")
            bt = backtest_simple(df_f, horizon=1, lookback_days=60)
            if bt:
                st.write(f"MAE: {bt['mae']:.4f}, RMSE: {bt['rmse']:.4f}, MAPE: {bt['mape']:.2f}% (n={bt['n']})")
            else:
                st.write("Backtest not available.")

# ---------------- Portfolio ----------------
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
        dmap = fetch_yf_batch(syms, period="7d", interval="1d", batch_size=6)
        for s in syms:
            d = dmap.get(s)
            if d is not None and 'Close' in d.columns and not d['Close'].dropna().empty:
                try:
                    price_map[s] = float(d['Close'].dropna().iloc[-1]) * fx
                except Exception:
                    price_map[s] = None
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

# ---------------- News ----------------
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

# ---------------- About ----------------
with tab_about:
    st.header("About")
    st.markdown("""
    **Global Crypto Insight** — polished crypto dashboard.
    - Top coins from CoinGecko, MovingAvg (improved) forecast by default, Prophet optional
    - Signals (MA/RSI/MACD), Trend heatmap, Fear & Greed gauge, Portfolio
    """)
    st.markdown("Run: `streamlit run app.py`")
    st.caption("Educational only — not financial advice.")

# Footer
st.markdown("---")
st.caption("Manage your own risk. For commercial release prepare README & LICENSE.")
