# app.py
# Crypto Preview (Persian UI) - initial visual + MA signals + short forecast
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Preview · نسخه اولیه", layout="wide")
st.title("🚀 داشبورد کریپتو — نسخهٔ اولیه (دیداری)")

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

# ---------- fetch data ----------
@st.cache_data(ttl=120)
def fetch_all(symbols_list, period_choice, interval):
    out = {}
    for s in symbols_list:
        df = safe_fetch(s, period_choice, interval)
        # normalize
        if df.empty:
            out[s] = pd.DataFrame()
            continue
        if np.issubdtype(df.index.dtype, np.datetime64):
            df = df.reset_index()
            if 'Datetime' in df.columns and 'Date' not in df.columns:
                df.rename(columns={'Datetime':'Date'}, inplace=True)
        if 'Date' not in df.columns:
            df.reset_index(inplace=True)
            df.rename(columns={df.columns[0]:'Date'}, inplace=True)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        if 'Close' not in df.columns:
            cands = [c for c in df.columns if 'close' in c.lower()]
            if cands:
                df.rename(columns={cands[0]:'Close'}, inplace=True)
        if 'Close' in df.columns:
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df = df.dropna(subset=['Date','Close']).sort_values('Date').reset_index(drop=True)
        out[s] = df
    return out

with st.spinner("در حال دریافت داده..."):
    data_map = fetch_all(symbols, period_choice, interval)

# ---------- summary table ----------
summary = []
for s in symbols:
    df = data_map.get(s, pd.DataFrame())
    if df.empty:
        summary.append({"Symbol": s, "Price": None, "Change24h(%)": None, "MA_Suggestion": "No data"})
        continue
    last_price = float(df['Close'].iloc[-1])
    # compute 24h change: try find ~24h ago row, else use previous row
    change24 = None
    try:
        ts = df['Date'].iloc[-1]
        target = ts - pd.Timedelta(days=1)
        prev = df[df['Date'] <= target]
        if not prev.empty:
            prev_val = float(prev['Close'].iloc[-1])
            change24 = (last_price - prev_val)/prev_val*100
        elif len(df) >= 2:
            prev_val = float(df['Close'].iloc[-2])
            change24 = (last_price - prev_val)/prev_val*100
    except Exception:
        change24 = None
    # MA suggestion
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    ma20 = df['MA20'].dropna()
    ma50 = df['MA50'].dropna()
    if not ma20.empty and not ma50.empty:
        ma20v = float(ma20.iloc[-1]); ma50v = float(ma50.iloc[-1])
        sug = "BUY" if ma20v > ma50v else "SELL" if ma20v < ma50v else "HOLD"
    elif not ma20.empty:
        ma20v = float(ma20.iloc[-1])
        sug = "BUY" if last_price > ma20v else "SELL" if last_price < ma20v else "HOLD"
    else:
        sug = "Unknown"
    summary.append({"Symbol": s, "Price": round(last_price,2), "Change24h(%)": round(change24,2) if change24 is not None else None, "MA_Suggestion": sug})

summary_df = pd.DataFrame(summary)

st.subheader("📋 خلاصه نمادها")
# simple color mapping with dataframe display
def fmt_price(v):
    return f"${v:,.2f}" if pd.notna(v) else "—"
def fmt_change(v):
    return f"{v:+.2f}%" if pd.notna(v) else "—"

st.dataframe(summary_df, use_container_width=True)

# ---------- detailed view for one symbol ----------
st.markdown("---")
st.subheader("🔎 جزئیات نماد")
chosen = st.selectbox("یک نماد برای بررسی دقیق‌تر انتخاب کنید:", symbols, index=0)
df = data_map.get(chosen, pd.DataFrame())
if df.empty:
    st.warning("داده‌ای برای این نماد موجود نیست.")
    st.stop()

# compute indicators
df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
if HAS_TA:
    try:
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        macd_obj = ta.trend.MACD(df['Close'])
        df['MACD_diff'] = macd_obj.macd_diff()
    except Exception:
        df['RSI'] = simple_rsi(df['Close'])
        _,_,df['MACD_diff'] = simple_macd(df['Close'])
else:
    df['RSI'] = simple_rsi(df['Close'])

# top metrics
last = float(df['Close'].iloc[-1])
st.metric(f"{chosen} — قیمت آخرین", f"${last:,.2f}")

# Forecast 3 & 7 days
fc3 = None; fc7 = None
if df.shape[0] >= 2 and HAS_PROPHET:
    try:
        prophet_df = df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=7, freq='D')
        pred = m.predict(future)
        tail = pred.tail(7)
        vals = tail['yhat'].values
        fc7 = vals
        fc3 = vals[:3] if len(vals)>=3 else vals
    except Exception:
        fc7 = moving_avg_forecast(df['Close'],7)
        fc3 = fc7[:3]
else:
    fc7 = moving_avg_forecast(df['Close'],7)
    fc3 = fc7[:3]

# show forecast table
fcdates = [(df['Date'].iloc[-1] + pd.Timedelta(days=i+1)).date() for i in range(len(fc7))]
fc_df = pd.DataFrame({"Date": fcdates, "Forecast": np.round(fc7,2)})
st.table(fc_df.head(7))

# plot candlestick + MA + forecast
fig = go.Figure()
if {'Open','High','Low','Close'}.issubset(df.columns):
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='OHLC'))
else:
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='gold')))
fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], mode='lines', name='MA50', line=dict(color='orange')))

start_date = df['Date'].iloc[-1] + pd.Timedelta(days=1)
dates7 = [start_date + pd.Timedelta(days=i) for i in range(len(fc7))]
fig.add_trace(go.Scatter(x=dates7, y=fc7, mode='lines+markers', name='Forecast7', line=dict(color='cyan', dash='dash')))

fig.update_layout(template='plotly_dark', height=520)
st.plotly_chart(fig, use_container_width=True)

# combined MA suggestion + RSI info
ma20v = float(df['MA20'].dropna().iloc[-1]) if not df['MA20'].dropna().empty else None
ma50v = float(df['MA50'].dropna().iloc[-1]) if not df['MA50'].dropna().empty else None
if ma20v is not None and ma50v is not None:
    if ma20v > ma50v:
        base_sig = "BUY"
    elif ma20v < ma50v:
        base_sig = "SELL"
    else:
        base_sig = "HOLD"
elif ma20v is not None:
    base_sig = "BUY" if last > ma20v else "SELL" if last < ma20v else "HOLD"
else:
    base_sig = "No MA"

# RSI hint
rsi_val = float(df['RSI'].dropna().iloc[-1]) if not df['RSI'].dropna().empty else None
rsi_hint = ""
if rsi_val is not None:
    if rsi_val < 30:
        rsi_hint = "RSI: Oversold"
    elif rsi_val > 70:
        rsi_hint = "RSI: Overbought"
    else:
        rsi_hint = f"RSI: {rsi_val:.1f}"

# final recommendation text
rec_text = f"Base MA signal: {base_sig}"
if rsi_hint:
    rec_text += f" · {rsi_hint}"
st.markdown(f"**پیشنهاد ترکیبی:** {rec_text}")

# raw data & download
st.markdown("---")
st.subheader("دادهٔ خام (آخرین ردیف‌ها)")
st.dataframe(df.tail(50))
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ دانلود CSV", csv, file_name=f"{chosen}_data_{datetime.now().strftime('%Y%m%d')}.csv", mime='text/csv')

st.caption("تذکر: این ابزار آموزشی است — تصمیمات سرمایه‌گذاری با ریسک همراه است.")
