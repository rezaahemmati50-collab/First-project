# app.py
# Crypto Premium — Final polished dashboard (Persian UI)
# Paste into app.py and run: streamlit run app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Premium · Final", layout="wide")

# optional libraries
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
    """Download from yfinance and return normalized DataFrame (may be empty)."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None:
            return pd.DataFrame()
        df = ensure_flat_columns(df)
        return df
    except Exception:
        return pd.DataFrame()

def safe_reset_index_to_date(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there's a Date column of datetime type."""
    if df.empty:
        return df
    try:
        # if index is datetime-like, reset it to column
        if pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.reset_index()
            if 'Datetime' in df.columns and 'Date' not in df.columns:
                df.rename(columns={'Datetime':'Date'}, inplace=True)
        else:
            # try to convert index to datetimes (coerce)
            idx_try = pd.to_datetime(df.index, errors='coerce')
            if idx_try.notna().sum() > 0:
                df = df.reset_index()
                df.rename(columns={df.columns[0]:'Date'}, inplace=True)
            else:
                df = df.reset_index()
                df.rename(columns={df.columns[0]:'Date'}, inplace=True)
    except Exception:
        df = df.reset_index()
        if df.columns[0].lower() not in ('date','datetime'):
            df.rename(columns={df.columns[0]:'Date'}, inplace=True)
    # normalize Date name
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

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

def compute_ma_signal(ma_short: float, ma_mid: float, ma_long: float = None):
    """Return (label, color_hex)"""
    try:
        if ma_long is not None:
            if ma_short > ma_mid > ma_long:
                return "STRONG BUY", "#00c853"
            if ma_short > ma_mid:
                return "BUY", "#43a047"
            if ma_short < ma_mid < ma_long:
                return "STRONG SELL", "#d50000"
            if ma_short < ma_mid:
                return "SELL", "#ff3d00"
            return "HOLD", "#9e9e9e"
        else:
            if ma_short > ma_mid:
                return "BUY", "#43a047"
            if ma_short < ma_mid:
                return "SELL", "#ff3d00"
            return "HOLD", "#9e9e9e"
    except Exception:
        return "UNKNOWN", "#9e9e9e"

# ---------------- UI Header ----------------
st.markdown("""<style>
h1 {text-align:center}
.header {text-align:center; color:#b9d6ff}
.card {background:#07111a; padding:12px; border-radius:8px;}
</style>""", unsafe_allow_html=True)

st.markdown("<h1>AureumPro · Crypto Premium</h1>", unsafe_allow_html=True)
st.markdown("<div class='header'>داشبورد پیشرفته — سیگنال MA + پیش‌بینی کوتاه‌مدت</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Controls ----------------
default_symbols = ["BTC-USD","ETH-USD","ADA-USD","SOL-USD","XRP-USD","DOGE-USD","DOT-USD","LTC-USD"]
col1, col2, col3 = st.columns([3,1,1])
with col1:
    symbols = st.multiselect("انتخاب نمادها (حداقل ۱):", default_symbols, default=["BTC-USD","ETH-USD"])
with col2:
    period_choice = st.selectbox("بازه (period):", ["1d","5d","7d","14d","1mo","3mo"], index=1)
with col3:
    if period_choice in ["1d","5d","7d"]:
        interval = st.selectbox("دقت (interval):", ["1m","5m","15m","60m"], index=1)
    else:
        interval = st.selectbox("دقت (interval):", ["1h","1d"], index=0)

if not symbols:
    st.warning("لطفاً حداقل یک نماد انتخاب کنید.")
    st.stop()

# ---------------- Fetch data (cached) ----------------
@st.cache_data(ttl=120)
def fetch_for_symbols(symbols_list, period_choice, interval):
    out = {}
    for s in symbols_list:
        df = safe_fetch(s, period_choice, interval)
        if df is None or df.empty:
            out[s] = pd.DataFrame()
            continue
        df = ensure_flat_columns(df)
        df = safe_reset_index_to_date(df)
        # find Close
        if 'Close' not in df.columns:
            candidates = [c for c in df.columns if 'close' in str(c).lower()]
            if candidates:
                df.rename(columns={candidates[0]:'Close'}, inplace=True)
        # numericize
        if 'Close' in df.columns:
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df = df.dropna(subset=['Date','Close']).sort_values('Date').reset_index(drop=True)
        out[s] = df
    return out

with st.spinner("در حال دریافت داده..."):
    data_map = fetch_for_symbols(symbols, period_choice, interval)

# ---------------- Build summary ----------------
summary = []
for s in symbols:
    df = data_map.get(s, pd.DataFrame())
    if df.empty:
        summary.append({"Symbol": s, "Price": None, "Change24h(%)": None, "MA20": None, "MA50": None, "MA200": None, "Signal": "No data"})
        continue

    last_price = float(df['Close'].iloc[-1])
    # change 24h (try timestamp approximate)
    change24 = None
    try:
        last_ts = df['Date'].iloc[-1]
        target = last_ts - pd.Timedelta(days=1)
        prev = df[df['Date'] <= target]
        if not prev.empty:
            prev_val = float(prev['Close'].iloc[-1])
            change24 = (last_price - prev_val) / prev_val * 100
        elif len(df) >= 2:
            prev_val = float(df['Close'].iloc[-2])
            change24 = (last_price - prev_val) / prev_val * 100
    except Exception:
        change24 = None

    # MAs
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['MA200'] = df['Close'].rolling(window=200, min_periods=1).mean()

    ma20 = float(df['MA20'].dropna().iloc[-1]) if not df['MA20'].dropna().empty else None
    ma50 = float(df['MA50'].dropna().iloc[-1]) if not df['MA50'].dropna().empty else None
    ma200 = float(df['MA200'].dropna().iloc[-1]) if not df['MA200'].dropna().empty else None

    sig_label, sig_color = ("No data", "#9e9e9e")
    if ma20 is None or ma50 is None:
        sig_label, sig_color = ("Insufficient MA", "#9e9e9e")
    else:
        sig_label, sig_color = compute_ma_signal(ma20, ma50, ma_long=ma200 if ma200 is not None else None)

    summary.append({
        "Symbol": s,
        "Price": last_price,
        "Change24h(%)": round(change24,2) if change24 is not None else None,
        "MA20": round(ma20,4) if ma20 is not None else None,
        "MA50": round(ma50,4) if ma50 is not None else None,
        "MA200": round(ma200,4) if ma200 is not None else None,
        "Signal": sig_label,
        "Color": sig_color
    })

summary_df = pd.DataFrame(summary)

# ---------------- Display summary with colors ----------------
st.subheader("📋 خلاصه نمادها")
# prepare display df
disp = summary_df.copy()
disp['Price'] = disp['Price'].apply(lambda v: f"${v:,.2f}" if pd.notna(v) else "—")
def fmt_ch(v):
    if pd.isna(v): return "—"
    return f"🔺{v:+.2f}%" if v>0 else f"🔻{v:+.2f}%"
disp['Change24h'] = disp['Change24h(%)'] = disp['Change24h(%)'].apply(lambda v: fmt_ch(v))

# show table (streamlit does not fully render Styler in all envs; use dataframe + colored cards below)
st.dataframe(disp[['Symbol','Price','Change24h(%)','MA20','MA50','MA200','Signal']].rename(columns={'Change24h(%)':'24h'}), use_container_width=True)

# show visual signal cards
cols = st.columns(len(summary_df))
for i, row in summary_df.iterrows():
    label = row['Symbol']
    sig = row['Signal']
    color = row['Color']
    price = f"${row['Price']:,.2f}" if pd.notna(row['Price']) else "—"
    ch = f"{row['Change24h(%)']:+.2f}%" if pd.notna(row['Change24h(%)']) else "—"
    with cols[i]:
        st.markdown(f"<div style='padding:10px;border-radius:8px;background:{color};color:#021014; text-align:center;'><strong>{label}</strong><br/>{sig}<br/><small>{price} · {ch}</small></div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- Detailed symbol view ----------------
st.subheader("🔎 نمای جزئی برای یک نماد")
chosen = st.selectbox("یک نماد انتخاب کنید:", symbols, index=0)
df_sym = data_map.get(chosen, pd.DataFrame())
if df_sym.empty:
    st.warning("داده‌ای برای نماد انتخاب شده موجود نیست.")
    st.stop()

# indicators
df_sym['MA20'] = df_sym['Close'].rolling(window=20, min_periods=1).mean()
df_sym['MA50'] = df_sym['Close'].rolling(window=50, min_periods=1).mean()
df_sym['MA200'] = df_sym['Close'].rolling(window=200, min_periods=1).mean()
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

last_price = float(df_sym['Close'].iloc[-1])
st.metric(label=f"{chosen} — آخرین قیمت", value=f"${last_price:,.2f}")

# Forecast (3 & 7 days)
fc7 = None
fc3 = None
if df_sym.shape[0] >= 2 and HAS_PROPHET:
    try:
        prophet_df = df_sym[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
        model = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=7, freq='D')
        pred = model.predict(future)
        tail = pred.tail(7)
        fc7 = tail['yhat'].values
        fc3 = fc7[:3] if len(fc7) >= 3 else fc7
    except Exception:
        fc7 = moving_avg_forecast(df_sym['Close'],7)
        fc3 = fc7[:3]
else:
    fc7 = moving_avg_forecast(df_sym['Close'],7)
    fc3 = fc7[:3]

# show forecast table
st.subheader("🔮 پیش‌بینی کوتاه‌مدت")
dates_fc = [(df_sym['Date'].iloc[-1] + pd.Timedelta(days=i+1)).date() for i in range(len(fc7))]
fc_table = pd.DataFrame({"Date": dates_fc, "Forecast (USD)": np.round(fc7,2)})
st.table(fc_table)

# plot price + MAs + forecast
fig = go.Figure()
if {'Open','High','Low','Close'}.issubset(df_sym.columns):
    fig.add_trace(go.Candlestick(x=df_sym['Date'], open=df_sym['Open'], high=df_sym['High'], low=df_sym['Low'], close=df_sym['Close'], name='OHLC'))
else:
    fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['Close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['MA20'], mode='lines', name='MA20', line=dict(color='gold')))
fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['MA50'], mode='lines', name='MA50', line=dict(color='orange')))
fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['MA200'], mode='lines', name='MA200', line=dict(color='violet')))

start_date = df_sym['Date'].iloc[-1] + pd.Timedelta(days=1)
dates7 = [start_date + pd.Timedelta(days=i) for i in range(len(fc7))]
fig.add_trace(go.Scatter(x=dates7, y=fc7, mode='lines+markers', name='Forecast 7d', line=dict(color='cyan', dash='dash')))

fig.update_layout(template='plotly_dark', height=520)
st.plotly_chart(fig, use_container_width=True)

# combined recommendation
ma20v = float(df_sym['MA20'].dropna().iloc[-1]) if not df_sym['MA20'].dropna().empty else None
ma50v = float(df_sym['MA50'].dropna().iloc[-1]) if not df_sym['MA50'].dropna().empty else None
ma200v = float(df_sym['MA200'].dropna().iloc[-1]) if not df_sym['MA200'].dropna().empty else None
sig_label, sig_color = compute_ma_signal(ma20v if ma20v is not None else 0,
                                         ma50v if ma50v is not None else 0,
                                         ma_long=ma200v if ma200v is not None else None)

rsi_val = float(df_sym['RSI'].dropna().iloc[-1]) if not df_sym['RSI'].dropna().empty else None
rsi_text = ""
if rsi_val is not None:
    if rsi_val < 30:
        rsi_text = " · RSI: Oversold"
    elif rsi_val > 70:
        rsi_text = " · RSI: Overbought"
    else:
        rsi_text = f" · RSI: {rsi_val:.1f}"

st.markdown(f"<div style='padding:16px;border-radius:10px;background:#071620;'><h2 style='color:{sig_color};margin:0;'>{sig_label}</h2><p style='color:#bdbdbd;margin-top:8px;'>توضیح: کراس‌اوور MA و شاخص‌های تکنیکال{rsi_text}</p></div>", unsafe_allow_html=True)

# indicators expander
with st.expander("📊 جزئیات اندیکاتورها (RSI / MACD)"):
    st.write("RSI (14):", round(float(df_sym['RSI'].dropna().iloc[-1]),2) if not df_sym['RSI'].dropna().empty else "نامشخص")
    if 'MACD_diff' in df_sym.columns:
        st.write("MACD diff:", round(float(df_sym['MACD_diff'].dropna().iloc[-1]),6) if not df_sym['MACD_diff'].dropna().empty else "نامشخص")
    st.line_chart(df_sym.set_index('Date')[['RSI']].dropna())

# data + download
st.markdown("---")
st.subheader("📄 دادهٔ خام و دانلود")
st.dataframe(df_sym.tail(100))
csv_sym = df_sym.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ دانلود CSV نماد", csv_sym, file_name=f"{chosen}_data_{datetime.now().strftime('%Y%m%d')}.csv", mime='text/csv')

st.caption("⚠️ این ابزار آموزشی است و توصیه سرمایه‌گذاری محسوب نمی‌شود. همواره مدیریت ریسک کنید.")
