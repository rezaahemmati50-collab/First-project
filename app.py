# app.py
# Crypto Preview - robust version (fixed index dtype checks)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Preview · Stable", layout="wide")
st.title("🚀 داشبورد کریپتو — نسخهٔ پایدار (Fixed Index Checks)")

# optional libs
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

# ---------- helpers ----------
def ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def safe_fetch(sym: str, period: str, interval: str) -> pd.DataFrame:
    """Fetch data robustly and flatten columns."""
    try:
        df = yf.download(sym, period=period, interval=interval, progress=False)
        if df is None:
            return pd.DataFrame()
        df = ensure_flat_columns(df)
        return df
    except Exception:
        return pd.DataFrame()

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

def moving_avg_forecast(series: pd.Series, days: int):
    if series.dropna().empty:
        return np.array([np.nan]*days)
    last = float(series.dropna().iloc[-1])
    avg_pct = series.pct_change().dropna().mean()
    if np.isnan(avg_pct):
        avg_pct = 0.0
    return np.array([ last * ((1+avg_pct)**i) for i in range(1, days+1) ])

# ---------- UI controls ----------
symbols_default = ["BTC-USD","ETH-USD","ADA-USD","XLM-USD","SOL-USD","DOGE-USD","DOT-USD"]
cols = st.columns([3,1,1])
with cols[0]:
    symbols = st.multiselect("انتخاب نمادها (حداقل ۱):", symbols_default, default=["BTC-USD","ETH-USD"])
with cols[1]:
    period_choice = st.selectbox("دوره داده:", ["1d","5d","7d","14d","1mo","3mo"], index=1)
with cols[2]:
    if period_choice in ["1d","5d","7d"]:
        interval = st.selectbox("دقت (interval):", ["1m","5m","15m","60m"], index=1)
    else:
        interval = st.selectbox("دقت (interval):", ["1h","1d"], index=0)

st.markdown("---")
if not symbols:
    st.warning("لطفاً حداقل یک نماد انتخاب کنید.")
    st.stop()

# ---------- fetch data (fixed index checks) ----------
@st.cache_data(ttl=120)
def fetch_all(symbols_list, period_choice, interval):
    out = {}
    for s in symbols_list:
        df = safe_fetch(s, period_choice, interval)
        if df is None or df.empty:
            out[s] = pd.DataFrame()
            continue

        # flatten columns
        df = ensure_flat_columns(df)

        # ---- robust index / Date handling ----
        # If index is datetime-like, reset to a Date column;
        # otherwise try to coerce index to datetime safely.
        try:
            # If index already datetime, move to column
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df = df.reset_index()
                # common names: Datetime -> Date
                if 'Datetime' in df.columns and 'Date' not in df.columns:
                    df.rename(columns={'Datetime':'Date'}, inplace=True)
            else:
                # try converting index values to datetime; if OK, set as Date column
                try:
                    idx_dt = pd.to_datetime(df.index, errors='coerce')
                    if idx_dt.notna().sum() > 0:
                        df = df.reset_index()
                        # ensure the reset column is named Date
                        if df.columns[0].lower() not in ('date','datetime'):
                            df.rename(columns={df.columns[0]:'Date'}, inplace=True)
                        else:
                            df.rename(columns={df.columns[0]:'Date'}, inplace=True)
                    else:
                        # fallback: leave df as is and later reset_index below
                        df = df.reset_index()
                        if df.columns[0].lower() not in ('date','datetime'):
                            df.rename(columns={df.columns[0]:'Date'}, inplace=True)
                        else:
                            df.rename(columns={df.columns[0]:'Date'}, inplace=True)
                except Exception:
                    df = df.reset_index()
                    if df.columns[0].lower() not in ('date','datetime'):
                        df.rename(columns={df.columns[0]:'Date'}, inplace=True)
                    else:
                        df.rename(columns={df.columns[0]:'Date'}, inplace=True)
        except Exception:
            # ultimate fallback
            df = df.reset_index()
            if df.columns[0].lower() not in ('date','datetime'):
                df.rename(columns={df.columns[0]:'Date'}, inplace=True)

        # normalize Date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        # find Close column or candidate
        if 'Close' not in df.columns:
            cands = [c for c in df.columns if 'close' in c.lower()]
            if cands:
                df.rename(columns={cands[0]:'Close'}, inplace=True)
        # numericize
        if 'Close' in df.columns:
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        # drop invalid rows
        df = df.dropna(subset=['Date','Close']).sort_values('Date').reset_index(drop=True)

        out[s] = df
    return out

with st.spinner("در حال دریافت داده برای نمادها..."):
    symbol_data = fetch_all(symbols, period_choice, interval)

# ---------- summary table ----------
summary_rows = []
for sym in symbols:
    df = symbol_data.get(sym, pd.DataFrame())
    if df.empty:
        summary_rows.append({"Symbol": sym, "Price": None, "Change24h(%)": None, "MA_Suggestion": "No data"})
        continue

    # latest price
    try:
        latest_price = float(df['Close'].iloc[-1])
    except Exception:
        latest_price = None

    # compute 24h change: try find ~24h ago, else previous value
    change_24h = None
    try:
        last_ts = df['Date'].iloc[-1]
        target = last_ts - pd.Timedelta(days=1)
        prev_row = df[df['Date'] <= target]
        if not prev_row.empty:
            prev_val = float(prev_row['Close'].iloc[-1])
            change_24h = (latest_price - prev_val) / prev_val * 100
        else:
            if len(df) >= 2:
                prev_val = float(df['Close'].iloc[-2])
                change_24h = (latest_price - prev_val) / prev_val * 100
    except Exception:
        change_24h = None

    # moving averages
    df['MA5'] = df['Close'].rolling(window=5, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()

    ma_short = float(df['MA20'].dropna().iloc[-1]) if not df['MA20'].dropna().empty else None
    ma_long = float(df['MA50'].dropna().iloc[-1]) if not df['MA50'].dropna().empty else None

    if (ma_short is not None) and (ma_long is not None):
        if ma_short > ma_long:
            sug = "BUY"
        elif ma_short < ma_long:
            sug = "SELL"
        else:
            sug = "HOLD"
    elif ma_short is not None:
        sug = "BUY" if (latest_price is not None and latest_price > ma_short) else "SELL" if (latest_price is not None and latest_price < ma_short) else "HOLD"
    else:
        sug = "Unknown"

    summary_rows.append({
        "Symbol": sym,
        "Price": round(latest_price,2) if latest_price is not None else None,
        "Change24h(%)": round(change_24h,2) if change_24h is not None else None,
        "MA_Suggestion": sug,
        "MA20": round(ma_short,4) if ma_short is not None else None,
        "MA50": round(ma_long,4) if ma_long is not None else None
    })

summary_df = pd.DataFrame(summary_rows)

# display table
st.subheader("📋 خلاصه نمادها")
st.dataframe(summary_df, use_container_width=True)

# ---------- detailed view for one symbol ----------
st.markdown("---")
st.subheader("🔎 نمای جزئی برای یک نماد")
sym_choice = st.selectbox("انتخاب نماد برای تحلیل:", symbols, index=0)
df_sym = symbol_data.get(sym_choice, pd.DataFrame())
if df_sym.empty:
    st.warning("داده‌ای برای این نماد در دسترس نیست.")
    st.stop()

# indicators for chosen symbol
df_sym['MA7'] = df_sym['Close'].rolling(window=7, min_periods=1).mean()
df_sym['MA20'] = df_sym['Close'].rolling(window=20, min_periods=1).mean()
df_sym['MA50'] = df_sym['Close'].rolling(window=50, min_periods=1).mean()
if HAS_TA:
    try:
        df_sym['RSI'] = ta.momentum.RSIIndicator(df_sym['Close'], window=14).rsi()
        macd_obj = ta.trend.MACD(df_sym['Close'])
        df_sym['MACD_diff'] = macd_obj.macd_diff()
    except Exception:
        df_sym['RSI'] = simple_rsi(df_sym['Close'])
        _,_,df_sym['MACD_diff'] = simple_macd(df_sym['Close'])
else:
    df_sym['RSI'] = simple_rsi(df_sym['Close'])
    _,_,df_sym['MACD_diff'] = simple_macd(df_sym['Close'])

# top metrics
last_price = float(df_sym['Close'].iloc[-1])
st.metric(label=f"{sym_choice} — قیمت آخرین", value=f"${last_price:,.2f}")

# Forecast 3 & 7 days (prophet if available)
if df_sym.shape[0] >= 2 and HAS_PROPHET:
    try:
        prophet_df = df_sym[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
        model = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=7, freq='D')
        pred = model.predict(future)
        tail = pred.tail(7)
        vals = tail['yhat'].values
        fc7 = vals
        fc3 = vals[:3] if len(vals)>=3 else vals
    except Exception:
        fc7 = moving_avg_forecast(df_sym['Close'],7)
        fc3 = fc7[:3]
else:
    fc7 = moving_avg_forecast(df_sym['Close'],7)
    fc3 = fc7[:3]

# show forecast table
fcdates = [(df_sym['Date'].iloc[-1] + pd.Timedelta(days=i+1)).date() for i in range(len(fc7))]
fc_df = pd.DataFrame({"Date": fcdates, "Forecast": np.round(fc7,2)})
st.table(fc_df.head(7))

# plot candlestick + MA + forecasts
fig = go.Figure()
if {'Open','High','Low','Close'}.issubset(df_sym.columns):
    fig.add_trace(go.Candlestick(x=df_sym['Date'], open=df_sym['Open'], high=df_sym['High'], low=df_sym['Low'], close=df_sym['Close'], name='OHLC'))
else:
    fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['MA20'], mode='lines', name='MA20', line=dict(color='gold')))
fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['MA50'], mode='lines', name='MA50', line=dict(color='orange')))

start_date = df_sym['Date'].iloc[-1] + pd.Timedelta(days=1)
dates7 = [start_date + pd.Timedelta(days=i) for i in range(len(fc7))]
fig.add_trace(go.Scatter(x=dates7, y=fc7, mode='lines+markers', name='Forecast7', line=dict(color='cyan', dash='dash')))

fig.update_layout(template='plotly_dark', height=520)
st.plotly_chart(fig, use_container_width=True)

# combined MA suggestion + RSI info
ma20v = float(df_sym['MA20'].dropna().iloc[-1]) if not df_sym['MA20'].dropna().empty else None
ma50v = float(df_sym['MA50'].dropna().iloc[-1]) if not df_sym['MA50'].dropna().empty else None
if ma20v is not None and ma50v is not None:
    if ma20v > ma50v:
        base_sig = "BUY"
    elif ma20v < ma50v:
        base_sig = "SELL"
    else:
        base_sig = "HOLD"
elif ma20v is not None:
    base_sig = "BUY" if last_price > ma20v else "SELL" if last_price < ma20v else "HOLD"
else:
    base_sig = "No MA"

# RSI hint
rsi_val = float(df_sym['RSI'].dropna().iloc[-1]) if not df_sym['RSI'].dropna().empty else None
rsi_hint = ""
if rsi_val is not None:
    if rsi_val < 30:
        rsi_hint = "RSI: Oversold"
    elif rsi_val > 70:
        rsi_hint = "RSI: Overbought"
    else:
        rsi_hint = f"RSI: {rsi_val:.1f}"

rec_text = f"Base MA signal: {base_sig}"
if rsi_hint:
    rec_text += f" · {rsi_hint}"
st.markdown(f"**پیشنهاد ترکیبی:** {rec_text}")

# raw data & download
st.markdown("---")
st.subheader("دادهٔ خام (آخرین ردیف‌ها)")
st.dataframe(df_sym.tail(50))
csv = df_sym.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ دانلود CSV", csv, file_name=f"{sym_choice}_data_{datetime.now().strftime('%Y%m%d')}.csv", mime='text/csv')

st.caption("تذکر: این ابزار آموزشی است — تصمیمات سرمایه‌گذاری با ریسک همراه است.")
