# app.py
# AureumPro · Global Edition
# Multi-tab crypto dashboard: Market | Forecast | Portfolio | News
# English UI, multi-currency support, robust fetching, prophet optional fallback.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
from io import StringIO

st.set_page_config(page_title="AureumPro · Global", layout="wide", initial_sidebar_state="expanded")

# ---------------- Optional libs flags ----------------
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
def ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def safe_fetch(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """Download data from yfinance; return empty DataFrame on problems"""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None:
            return pd.DataFrame()
        df = ensure_flat_columns(df)
        return df
    except Exception:
        return pd.DataFrame()

def reset_index_to_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        if pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.reset_index()
            if 'Datetime' in df.columns and 'Date' not in df.columns:
                df.rename(columns={'Datetime':'Date'}, inplace=True)
        else:
            df = df.reset_index()
            df.rename(columns={df.columns[0]:'Date'}, inplace=True)
    except Exception:
        df = df.reset_index()
        df.rename(columns={df.columns[0]:'Date'}, inplace=True)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

def moving_avg_forecast(series: pd.Series, days: int):
    if series.dropna().empty:
        return np.array([np.nan]*days)
    last = float(series.dropna().iloc[-1])
    avg_pct = series.pct_change().dropna().mean() if series.pct_change().dropna().shape[0] > 0 else 0.0
    return np.array([ last * ((1+avg_pct)**i) for i in range(1, days+1) ])

def simple_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))

def simple_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff

def compute_signal_combined(close_series: pd.Series, next_forecast: float = None):
    """Return (label, color_hex, reasons). close_series: pd.Series of Close prices"""
    if close_series.dropna().empty:
        return ("No data", "#9e9e9e", "Insufficient data")
    # MAs
    ma20 = close_series.rolling(window=20, min_periods=1).mean().iloc[-1]
    ma50 = close_series.rolling(window=50, min_periods=1).mean().iloc[-1]
    ma200 = close_series.rolling(window=200, min_periods=1).mean().iloc[-1]
    # RSI / MACD
    try:
        if HAS_TA:
            rsi = ta.momentum.RSIIndicator(close_series, window=14).rsi().iloc[-1]
            macd_diff = ta.trend.MACD(close_series).macd_diff().iloc[-1]
        else:
            rsi = simple_rsi(close_series).iloc[-1]
            _,_,macd = simple_macd(close_series)
            macd_diff = macd.iloc[-1]
    except Exception:
        rsi = np.nan
        macd_diff = np.nan

    score = 0
    reasons = []
    # MA rules
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
        if rsi < 30:
            score += 1; reasons.append(f"RSI low ({rsi:.1f})")
        elif rsi > 70:
            score -= 1; reasons.append(f"RSI high ({rsi:.1f})")
    except Exception:
        pass
    # MACD
    try:
        if macd_diff > 0:
            score += 1; reasons.append("MACD positive")
        else:
            score -= 1; reasons.append("MACD negative")
    except Exception:
        pass
    # forecast influence
    try:
        if next_forecast is not None and not np.isnan(next_forecast):
            last = float(close_series.iloc[-1])
            pct = (next_forecast - last) / last
            if pct > 0.01:
                score += 1; reasons.append(f"Forecast +{pct*100:.2f}%")
            elif pct < -0.01:
                score -= 1; reasons.append(f"Forecast {pct*100:.2f}%")
    except Exception:
        pass
    # label by score
    if score >= 4:
        return ("STRONG BUY", "#00c853", " · ".join(reasons))
    if score >= 2:
        return ("BUY", "#43a047", " · ".join(reasons))
    if score == 1:
        return ("MILD BUY", "#f9a825", " · ".join(reasons))
    if score == 0:
        return ("HOLD", "#9e9e9e", " · ".join(reasons))
    if score == -1:
        return ("MILD SELL", "#ff6d00", " · ".join(reasons))
    return ("SELL", "#d50000", " · ".join(reasons))

# ---------------- Sidebar settings ----------------
st.sidebar.header("AureumPro Settings")
currency = st.sidebar.selectbox("Display currency", ["USD","CAD","EUR","GBP"], index=0)
theme = st.sidebar.selectbox("Theme", ["Dark","Light"], index=0)
if theme == "Dark":
    pt_template = "plotly_dark"
else:
    pt_template = "plotly"

# exchange conversion helper (USD -> chosen)
@st.cache_data(ttl=300)
def get_fx_rate(target_currency: str):
    # attempt: use yfinance ticker USDCAD=X, etc (USD->TARGET)
    if target_currency == "USD":
        return 1.0
    ticker_map = {"CAD":"USDCAD=X","EUR":"USDEUR=X","GBP":"USDGBP=X"}
    t = ticker_map.get(target_currency, None)
    if t is None:
        return 1.0
    try:
        df = yf.download(t, period="1d", interval="1d", progress=False)
        if df is None or df.empty:
            return 1.0
        last = df['Close'].dropna().iloc[-1]
        return float(last)
    except Exception:
        return 1.0

fx_rate = get_fx_rate(currency)

# ---------------- Tabs ----------------
tabs = st.tabs(["Market", "Forecast", "Portfolio", "News", "About"])
tab_market, tab_forecast, tab_portfolio, tab_news, tab_about = tabs

# ---------------- Market Tab ----------------
with tab_market:
    st.header("Live Market")
    # default list + manual add + search
    default_symbols = ["BTC-USD","ETH-USD","ADA-USD","SOL-USD","XRP-USD","DOGE-USD","DOT-USD","LTC-USD"]
    colA, colB = st.columns([3,2])
    with colA:
        add_symbol = st.text_input("Add symbol (e.g. ADA-USD) and press Enter", value="")
        time_period = st.selectbox("Period", ["7d","14d","1mo","3mo"], index=0)
    with colB:
        search = st.text_input("Search symbol or name (filter table)", value="")

    # build symbols list
    symbols = default_symbols.copy()
    if add_symbol:
        add_symbol = add_symbol.strip().upper()
        if add_symbol not in symbols:
            symbols.insert(0, add_symbol)

    # fetch data for summary
    @st.cache_data(ttl=90)
    def fetch_summary(symbols_list, period, interval="1h"):
        out = {}
        for s in symbols_list:
            df = safe_fetch(s, period, interval)
            if df is None or df.empty:
                out[s] = pd.DataFrame()
                continue
            df = ensure_flat_columns(df)
            df = reset_index_to_date(df)
            if 'Close' not in df.columns:
                cands = [c for c in df.columns if 'close' in str(c).lower()]
                if cands:
                    df.rename(columns={cands[0]:'Close'}, inplace=True)
            if 'Close' in df.columns:
                df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Date','Close']).sort_values('Date').reset_index(drop=True)
            out[s] = df
        return out

    with st.spinner("Fetching market data..."):
        data_map = fetch_summary(symbols, time_period, interval="1h")

    # build summary rows
    rows = []
    for s in symbols:
        df = data_map.get(s, pd.DataFrame())
        if df.empty:
            rows.append({"symbol":s,"price":None,"change24":None,"sig":"No data","color":"#9e9e9e"})
            continue
        price_usd = float(df['Close'].iloc[-1])
        price = price_usd * fx_rate
        # change 24h
        change24 = None
        try:
            last_ts = df['Date'].iloc[-1]
            prev = df[df['Date'] <= (last_ts - pd.Timedelta(days=1))]
            if not prev.empty:
                prev_val = float(prev['Close'].iloc[-1])
            else:
                prev_val = float(df['Close'].iloc[-2]) if len(df)>=2 else price_usd
            change24 = (price_usd - prev_val)/prev_val*100
        except Exception:
            change24 = None
        # forecast 3-day fallback
        fc3 = moving_avg_forecast(df['Close'],3)
        fc3_val = float(fc3[0]) if len(fc3)>0 else None
        sig, col, reason = compute_signal_combined(df['Close'], next_forecast=fc3_val)
        rows.append({
            "symbol": s,
            "price": price,
            "change24": round(change24,2) if change24 is not None else None,
            "sig": sig,
            "color": col,
            "reason": reason
        })

    summary_df = pd.DataFrame(rows)

    # filter by search
    if search:
        q = search.strip().lower()
        summary_df = summary_df[ summary_df['symbol'].str.lower().str.contains(q) | summary_df['sig'].str.lower().str.contains(q) ]

    # format price column to have 2 decimals
    def fmt_cur(x):
        try:
            return f"{x:,.2f} {currency}"
        except Exception:
            return "—"
    summary_df['price_str'] = summary_df['price'].apply(lambda v: fmt_cur(v) if v is not None else "—")
    summary_df['change_str'] = summary_df['change24'].apply(lambda v: f"{v:+.2f}%" if (v is not None) else "—")

    st.dataframe(summary_df[['symbol','price_str','change_str','sig']].rename(columns={
        "symbol":"Symbol","price_str":"Price","change_str":"24h","sig":"Signal"
    }), use_container_width=True)

    # signal cards
    st.markdown("### Signal Cards")
    cols = st.columns(min(6, len(summary_df)))
    for i, r in summary_df.iterrows():
        c = cols[i % len(cols)]
        with c:
            color = r['color']
            st.markdown(f"<div style='background:{color};padding:10px;border-radius:8px;color:#021014;text-align:center;'><strong>{r['symbol']}</strong><br/>{r['sig']}<br/>{r['price_str']} · {r['change_str']}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.info("Click Forecast tab to run multi-day forecasts and compare models.")

# ---------------- Forecast Tab ----------------
with tab_forecast:
    st.header("Forecast & Backtest")
    # symbol selector
    default_for_symbol = "ADA-USD"
    f_sym = st.selectbox("Select symbol for forecasting", options=symbols, index= symbols.index(default_for_symbol) if default_for_symbol in symbols else 0)
    f_period = st.selectbox("Historical period for modeling", ["1mo","3mo","6mo","1y"], index=1)
    f_interval = st.selectbox("Interval for historical data", ["1d","1h"], index=0)
    f_model = st.selectbox("Model", ["Prophet (if installed)","MovingAvg (fallback)"], index=0)
    f_horizon = st.selectbox("Forecast horizon (days)", [3,7,30], index=1)

    @st.cache_data(ttl=300)
    def fetch_df(sym, period, interval):
        df = safe_fetch(sym, period, interval)
        if df is None or df.empty:
            return pd.DataFrame()
        df = ensure_flat_columns(df)
        df = reset_index_to_date(df)
        if 'Close' not in df.columns:
            cands = [c for c in df.columns if 'close' in str(c).lower()]
            if cands:
                df.rename(columns={cands[0]:'Close'}, inplace=True)
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Date','Close']).sort_values('Date').reset_index(drop=True)
        return df

    with st.spinner("Loading history..."):
        df_fore = fetch_df(f_sym, f_period, f_interval)

    if df_fore.empty:
        st.warning("No historical data available for this symbol/period.")
    else:
        st.subheader(f"Historical: {f_sym} ({f_period})")
        st.line_chart(df_fore.set_index('Date')['Close'])

        # Build forecast
        forecast_vals = None
        try:
            if f_model.startswith("Prophet") and HAS_PROPHET and df_fore.shape[0] >= 3:
                prophet_df = df_fore[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
                m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
                m.fit(prophet_df)
                future = m.make_future_dataframe(periods=f_horizon, freq='D')
                pred = m.predict(future)
                tail = pred.tail(f_horizon)
                forecast_vals = tail['yhat'].values
            else:
                forecast_vals = moving_avg_forecast(df_fore['Close'], f_horizon)
        except Exception as e:
            st.error(f"Forecast error: {e}")
            forecast_vals = moving_avg_forecast(df_fore['Close'], f_horizon)

        # present forecast table & plot with history
        last_date = df_fore['Date'].iloc[-1]
        dates_fc = [ (last_date + timedelta(days=i+1)).date() for i in range(len(forecast_vals)) ]
        fc_df = pd.DataFrame({"Date": dates_fc, "Forecast": np.round(forecast_vals,4)})
        st.subheader("Forecast")
        st.table(fc_df)

        # plot history + forecast
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_fore['Date'], y=df_fore['Close'], mode='lines', name='History'))
        fig.add_trace(go.Scatter(x=dates_fc, y=forecast_vals, mode='lines+markers', name=f'Forecast {f_horizon}d', line=dict(dash='dash')))
        fig.update_layout(template=pt_template, height=520)
        st.plotly_chart(fig, use_container_width=True)

        # backtest quick (rolling origin) for last 60 days: compute MAE/RMSE/MAPE
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        def backtest_simple(df_hist, model='ma', horizon=3, lookback_days=60):
            df = df_hist[['Date','Close']].set_index('Date').copy()
            df = df.asfreq('D', method='pad').dropna()  # make daily
            n = len(df)
            preds=[]; truths=[]
            dates=[]
            for i in range(max(0,n - lookback_days)):
                train = df.iloc[:i+1]['Close']
                if train.shape[0] < 5:
                    continue
                # predict next day (horizon=1 for simplicity)
                if model=='prophet' and HAS_PROPHET and train.shape[0]>=10:
                    tr = train.reset_index().rename(columns={'Date':'ds','Close':'y'})
                    try:
                        mm = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
                        mm.fit(tr)
                        fut = mm.make_future_dataframe(periods=horizon, freq='D')
                        p = mm.predict(fut)
                        yhat = p['yhat'].values[-1]
                    except Exception:
                        yhat = train.iloc[-1] * (1 + train.pct_change().tail(7).mean())
                else:
                    yhat = train.iloc[-1] * (1 + train.pct_change().tail(7).mean())
                actual_idx = train.index[-1] + timedelta(days=1)
                if actual_idx in df.index:
                    actual = df.loc[actual_idx]['Close']
                    preds.append(yhat); truths.append(actual); dates.append(actual_idx)
            if len(truths)==0:
                return None
            preds=np.array(preds); truths=np.array(truths)
            mae = mean_absolute_error(truths, preds)
            rmse = mean_squared_error(truths, preds, squared=False)
            mape = np.mean(np.abs((truths-preds)/truths))*100
            return {"mae":mae,"rmse":rmse,"mape":mape, "n":len(truths)}

        st.subheader("Quick Backtest (rolling)")
        bt = backtest_simple(df_fore, model='prophet' if f_model.startswith("Prophet") else 'ma', horizon=1, lookback_days=60)
        if bt is None:
            st.write("Backtest too small (not enough daily data).")
        else:
            st.write(f"MAE: {bt['mae']:.4f}, RMSE: {bt['rmse']:.4f}, MAPE: {bt['mape']:.2f}% (n={bt['n']})")

# ---------------- Portfolio Tab ----------------
with tab_portfolio:
    st.header("Portfolio")
    # simple portfolio in session_state
    if 'portfolio' not in st.session_state:
        st.session_state['portfolio'] = []  # list of dicts {symbol, qty, cost_currency, cost_per_unit}

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        p_sym = st.text_input("Symbol to add (e.g. ADA-USD)", value="")
    with col2:
        p_qty = st.number_input("Quantity", min_value=0.0, value=0.0, step=0.1)
    with col3:
        p_cost = st.number_input(f"Cost per unit ({currency})", min_value=0.0, value=0.0, step=0.01)

    if st.button("Add to portfolio"):
        s = p_sym.strip().upper()
        if s=="" or p_qty<=0 or p_cost<=0:
            st.warning("Provide symbol, qty>0 and cost>0.")
        else:
            st.session_state['portfolio'].append({"symbol":s,"qty":p_qty,"cost_per_unit":p_cost,"added":datetime.now().isoformat()})
            st.success(f"Added {p_qty} x {s} @ {p_cost} {currency}")

    st.subheader("Holdings")
    port = st.session_state['portfolio']
    if len(port)==0:
        st.info("Your portfolio is empty. Add holdings above.")
    else:
        # fetch latest prices for portfolio symbols
        port_syms = list({p['symbol'] for p in port})
        prices_map = {}
        for s in port_syms:
            dfp = safe_fetch(s, period="7d", interval="1d")
            if not dfp.empty:
                dfp = ensure_flat_columns(dfp)
                dfp = reset_index_to_date(dfp)
                if 'Close' in dfp.columns and not dfp['Close'].dropna().empty:
                    prices_map[s] = float(dfp['Close'].iloc[-1]) * fx_rate
                else:
                    prices_map[s] = None
            else:
                prices_map[s] = None
        # display table
        rows=[]
        for p in port:
            current_price = prices_map.get(p['symbol'], None)
            value = current_price * p['qty'] if current_price is not None else None
            cost_total = p['cost_per_unit'] * p['qty']
            pnl = value - cost_total if value is not None else None
            rows.append({"symbol":p['symbol'],"qty":p['qty'],"cost_per_unit":p['cost_per_unit'],"current_price":current_price,"value":value,"pnl":pnl})
        dfp = pd.DataFrame(rows)
        def fmt_val(x):
            return f"{x:,.2f} {currency}" if (x is not None and not pd.isna(x)) else "—"
        dfp['current_price'] = dfp['current_price'].apply(fmt_val)
        dfp['value'] = dfp['value'].apply(fmt_val)
        dfp['pnl'] = dfp['pnl'].apply(lambda v: f"{v:,.2f} {currency}" if (v is not None and not pd.isna(v)) else "—")
        st.table(dfp)

# ---------------- News Tab ----------------
with tab_news:
    st.header("News")
    st.markdown("You can paste an RSS feed URL (e.g. CoinDesk RSS) or try automatic fetch. Internet required.")
    rss_url = st.text_input("RSS feed URL (optional)", value="")
    if st.button("Fetch news"):
        if not rss_url:
            st.info("No RSS provided — trying CoinDesk headlines via public RSS.")
            rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
        try:
            r = requests.get(rss_url, timeout=10)
            if r.status_code == 200:
                txt = r.text
                # very basic parse for <title> tags
                titles = []
                parts = txt.split("<title>")
                for p in parts[1:6]:  # first few
                    t = p.split("</title>")[0]
                    titles.append(t)
                st.write("Top headlines:")
                for t in titles:
                    st.write("-", t)
            else:
                st.error("Could not fetch RSS — status: " + str(r.status_code))
        except Exception as e:
            st.error("News fetch failed: " + str(e))

# ---------------- About Tab ----------------
with tab_about:
    st.header("About AureumPro")
    st.markdown("""
    **AureumPro** — polished multi-tab crypto dashboard.
    Features:
    - Market watch, Forecasting (Prophet optional), Portfolio.
    - Multi-currency display (USD, CAD, EUR, GBP).
    - Signal combining MA/RSI/MACD + forecast influence.
    - Export CSV & quick backtests.
    """)
    st.markdown("**Run**: `streamlit run app.py`")
    st.markdown("**Requirements**: see requirements.txt (prophet/ta optional)")

# ---------------- End ----------------
