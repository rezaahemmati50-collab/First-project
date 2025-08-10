# app.py
# Crypto All-in-One Dashboard — Persian UI
# Features:
# - Two-decimal price formatting
# - Search bar + manual symbol add
# - MA / RSI / MACD based signals
# - Forecast 3 & 7 days (prophet optional; fallback: moving average)
# - Download CSV, candlestick/forecast plot, colored signal cards

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Complete · داشبورد", layout="wide")

# --- optional packages ---
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

# ---------------- helpers ----------------
def ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def safe_fetch(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None:
            return pd.DataFrame()
        return ensure_flat_columns(df)
    except Exception:
        return pd.DataFrame()

def safe_reset_index_to_date(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        if pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.reset_index()
            if 'Datetime' in df.columns and 'Date' not in df.columns:
                df.rename(columns={'Datetime':'Date'}, inplace=True)
        else:
            idx_try = pd.to_datetime(df.index, errors='coerce')
            df = df.reset_index()
            df.rename(columns={df.columns[0]:'Date'}, inplace=True)
    except Exception:
        df = df.reset_index()
        df.rename(columns={df.columns[0]:'Date'}, inplace=True)
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
    avg_pct = series.pct_change().dropna().mean() if series.pct_change().dropna().shape[0]>0 else 0.0
    return np.array([ last * ((1+avg_pct)**i) for i in range(1, days+1) ])

def compute_signal_combined(df: pd.DataFrame, next_forecast: float = None):
    # returns (label, color_hex, reason_text)
    if df.empty or 'Close' not in df.columns:
        return ("No data", "#9e9e9e", "داده ناکافی")
    close = df['Close'].dropna()
    if close.empty:
        return ("No data", "#9e9e9e", "دادهٔ بسته")
    # compute MAs
    ma20 = close.rolling(window=20, min_periods=1).mean().iloc[-1]
    ma50 = close.rolling(window=50, min_periods=1).mean().iloc[-1]
    ma200 = close.rolling(window=200, min_periods=1).mean().iloc[-1]
    # RSI/MACD
    try:
        if HAS_TA:
            rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
            macd_diff = ta.trend.MACD(close).macd_diff().iloc[-1]
        else:
            rsi = simple_rsi(close).iloc[-1]
            _,_,macd_diff = simple_macd(close)
            macd_diff = macd_diff.iloc[-1]
    except Exception:
        rsi = np.nan
        macd_diff = np.nan

    score = 0
    reasons = []
    # MA logic
    try:
        if ma20 > ma50:
            score += 2; reasons.append("MA20>MA50")
        elif ma20 < ma50:
            score -= 2; reasons.append("MA20<MA50")
        if not np.isnan(ma200):
            if ma50 > ma200:
                score += 1; reasons.append("MA50>MA200")
            else:
                score -= 1; reasons.append("MA50<MA200")
    except Exception:
        pass
    # RSI
    try:
        if rsi < 30:
            score += 1; reasons.append(f"RSI پایین ({rsi:.1f})")
        elif rsi > 70:
            score -= 1; reasons.append(f"RSI بالا ({rsi:.1f})")
    except Exception:
        pass
    # MACD
    try:
        if macd_diff > 0:
            score += 1; reasons.append("MACD مثبت")
        elif macd_diff < 0:
            score -= 1; reasons.append("MACD منفی")
    except Exception:
        pass
    # forecast influence
    try:
        if next_forecast is not None and not np.isnan(next_forecast):
            last = float(close.iloc[-1])
            pct = (next_forecast - last) / last
            if pct > 0.01:
                score += 1; reasons.append(f"پیش‌بینی +{pct*100:.2f}%")
            elif pct < -0.01:
                score -= 1; reasons.append(f"پیش‌بینی {pct*100:.2f}%")
    except Exception:
        pass

    # label by score
    if score >= 3:
        return ("🔵 سیگنال قوی خرید", "#00c853", " · ".join(reasons))
    if score == 2:
        return ("🟢 سیگنال خرید", "#43a047", " · ".join(reasons))
    if score == 1:
        return ("🟡 تمایل به خرید", "#f9a825", " · ".join(reasons))
    if score == 0:
        return ("⚪ نگه‌داری", "#9e9e9e", " · ".join(reasons))
    if score == -1:
        return ("🟠 سیگنال فروش", "#ff6d00", " · ".join(reasons))
    return ("🔴 سیگنال قوی فروش", "#d50000", " · ".join(reasons))

# ---------------- UI header ----------------
st.markdown("<h1 style='text-align:center;'>AureumPro · Crypto Complete</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#9aa0a6;'>قیمت‌ها، سیگنال و پیش‌بینی — نسخهٔ کامل</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- Controls: default list + manual add + search ----------------
default_symbols = ["BTC-USD","ETH-USD","ADA-USD","SOL-USD","XRP-USD","DOGE-USD","DOT-USD","LTC-USD"]
col_a, col_b = st.columns([3,2])
with col_a:
    user_add = st.text_input("اضافه کردن نماد دستی (مثال: ADA-USD). ثبت با Enter:", value="")
    if user_add:
        user_add = user_add.strip().upper()
with col_b:
    search_input = st.text_input("جستجو (نماد یا نام؛ فیلتر جدول):", value="")

# maintain list (merge defaults + manual)
symbols = default_symbols.copy()
if user_add:
    if user_add not in symbols:
        symbols.insert(0, user_add)  # manual on top

# period/interval
col1, col2 = st.columns([1,1])
with col1:
    period_choice = st.selectbox("بازه (period):", ["7d","14d","1mo","3mo"], index=0)
with col2:
    interval = st.selectbox("دقت (interval):", ["1h","4h","1d"], index=0)

st.markdown("---")

# ---------------- Fetch data cached ----------------
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
        # find Close col
        if 'Close' not in df.columns:
            cands = [c for c in df.columns if 'close' in str(c).lower()]
            if cands:
                df.rename(columns={cands[0]:'Close'}, inplace=True)
        # numericize
        if 'Close' in df.columns:
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df = df.dropna(subset=['Date','Close']).sort_values('Date').reset_index(drop=True)
        out[s] = df
    return out

with st.spinner("دریافت داده‌ها..."):
    data_map = fetch_for_symbols(symbols, period_choice, interval)

# ---------------- Build summary rows ----------------
rows = []
for s in symbols:
    df = data_map.get(s, pd.DataFrame())
    if df.empty:
        rows.append({
            "Symbol": s,
            "Price": None,
            "Change24h": None,
            "MA20": None,
            "MA50": None,
            "Signal": "No data",
            "SignalColor": "#9e9e9e",
            "Forecast3": None,
            "Forecast7": None,
            "Reason": ""
        })
        continue

    last_price = float(df['Close'].iloc[-1])
    # change 24h try by timestamp otherwise previous row
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

    # Forecast
    if df.shape[0] >= 3 and HAS_PROPHET:
        try:
            prophet_df = df[['Date','Close']].rename(columns={'Date':'ds','Close':'y'})
            m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=7, freq='D')
            pred = m.predict(future)
            vals = pred['yhat'].values[-7:]
            fc7 = vals
            fc3 = vals[:3] if len(vals)>=3 else vals
        except Exception:
            fc7 = moving_avg_forecast(df['Close'],7)
            fc3 = fc7[:3]
    else:
        fc7 = moving_avg_forecast(df['Close'],7)
        fc3 = fc7[:3]

    next_forecast = float(fc3[0]) if (fc3 is not None and len(fc3)>0 and not np.isnan(fc3[0])) else None

    label, color, reason = compute_signal_combined(df, next_forecast)

    rows.append({
        "Symbol": s,
        "Price": last_price,
        "Change24h": round(change24,2) if change24 is not None else None,
        "MA20": round(df['MA20'].iloc[-1],6) if 'MA20' in df.columns else None,
        "MA50": round(df['MA50'].iloc[-1],6) if 'MA50' in df.columns else None,
        "Signal": label,
        "SignalColor": color,
        "Forecast3": round(float(fc3[0]),4) if (fc3 is not None and len(fc3)>0 and not np.isnan(fc3[0])) else None,
        "Forecast7": [round(float(x),4) if not np.isnan(x) else None for x in fc7],
        "Reason": reason
    })

summary_df = pd.DataFrame(rows)

# ---------------- Search/filter ----------------
if search_input:
    q = search_input.strip().lower()
    mask = summary_df['Symbol'].str.lower().str.contains(q) | summary_df['Signal'].str.lower().str.contains(q)
    display_df = summary_df[mask].copy()
else:
    display_df = summary_df.copy()

# format numbers: Price with 2 decimals always
def fmt_price(x):
    try:
        if pd.isna(x) or x is None:
            return "—"
        return f"{x:,.2f}"
    except Exception:
        return str(x)

display_df['Price'] = display_df['Price'].apply(fmt_price)
display_df['Change24h'] = display_df['Change24h'].apply(lambda v: f"{v:+.2f}%" if (v is not None and not pd.isna(v)) else "—")
display_df['Forecast3'] = display_df['Forecast3'].apply(lambda v: f"{v:.2f}" if (v is not None and not pd.isna(v)) else "—")
display_df['Signal'] = display_df['Signal'].astype(str)

# ---------------- Table display ----------------
st.subheader("📋 خلاصه نمادها")
st.markdown("قیمت‌ها با دو رقم اعشار نمایش داده شده‌اند. برای مشاهده جزئیات یک نماد آن را انتخاب کنید.")
st.dataframe(display_df[['Symbol','Price','Change24h','MA20','MA50','Signal','Forecast3']].rename(columns={
    'Symbol':'نماد',
    'Price':'قیمت (USD)',
    'Change24h':'تغییر ۲۴ساعته',
    'MA20':'MA20',
    'MA50':'MA50',
    'Signal':'سیگنال',
    'Forecast3':'پیش‌بینی ۳ روزه'
}), use_container_width=True)

# colored signal cards (compact)
st.markdown("### کارت‌های سیگنال")
cards_cols = st.columns(min(6, len(display_df)))
for i, (_, r) in enumerate(display_df.iterrows()):
    col = cards_cols[i % len(cards_cols)]
    symbol = r['Symbol']
    sig = r['Signal']
    color = summary_df.loc[summary_df['Symbol']==symbol, 'SignalColor'].values[0]
    price_str = display_df.loc[display_df['Symbol']==symbol, 'Price'].values[0]
    change = display_df.loc[display_df['Symbol']==symbol, 'Change24h'].values[0]
    with col:
        st.markdown(f"<div style='background:{color};padding:10px;border-radius:8px;text-align:center;color:#021014;'><strong>{symbol}</strong><br/>{sig}<br/>{price_str} · {change}</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- Detailed view for one symbol ----------------
st.subheader("🔎 نمای جزئی برای یک نماد")
chosen = st.selectbox("یک نماد را انتخاب کنید:", display_df['Symbol'].tolist(), index=0)
df_sym = data_map.get(chosen, pd.DataFrame())
if df_sym.empty:
    st.warning("داده‌ای برای نماد انتخاب شده موجود نیست.")
else:
    # indicators
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

    last_price = float(df_sym['Close'].iloc[-1])
    st.metric(label=f"{chosen} — آخرین قیمت", value=f"${last_price:,.2f}")

    # Forecast display (from summary_df Forecast7 saved earlier)
    fc7_list = summary_df.loc[summary_df['Symbol']==chosen, 'Forecast7'].values
    fc7 = fc7_list[0] if len(fc7_list)>0 else None
    if fc7 is None:
        fc7 = moving_avg_forecast(df_sym['Close'],7)
    # show forecast table
    dates_fc = [(df_sym['Date'].iloc[-1] + pd.Timedelta(days=i+1)).date() for i in range(len(fc7))]
    fc_table = pd.DataFrame({"Date": dates_fc, "Forecast (USD)": [f"{x:.2f}" if x is not None else "—" for x in fc7]})
    st.subheader("🔮 پیش‌بینی ۷ روزه")
    st.table(fc_table)

    # plot
    fig = go.Figure()
    if {'Open','High','Low','Close'}.issubset(df_sym.columns):
        fig.add_trace(go.Candlestick(x=df_sym['Date'], open=df_sym['Open'], high=df_sym['High'], low=df_sym['Low'], close=df_sym['Close'], name='OHLC'))
    else:
        fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['MA20'], mode='lines', name='MA20', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df_sym['Date'], y=df_sym['MA50'], mode='lines', name='MA50', line=dict(color='orange')))

    start_date = df_sym['Date'].iloc[-1] + pd.Timedelta(days=1)
    dates7 = [start_date + pd.Timedelta(days=i) for i in range(len(fc7))]
    fig.add_trace(go.Scatter(x=dates7, y=fc7, mode='lines+markers', name='Forecast 7d', line=dict(color='cyan', dash='dash')))
    fig.update_layout(template='plotly_dark', height=520)
    st.plotly_chart(fig, use_container_width=True)

    # show reason & indicators
    label, color_code, reason = compute_signal_combined(df_sym, next_forecast=float(fc7[0]) if (fc7 is not None and len(fc7)>0) else None)
    st.markdown(f"<div style='padding:12px;border-radius:8px;background:#071620;'><h3 style='color:{color_code};margin:0;'>{label}</h3><p style='color:#bdbdbd;margin-top:8px;'>{reason}</p></div>", unsafe_allow_html=True)
    with st.expander("📊 اندیکاتورها (جزئیات)"):
        rsi_val = df_sym['RSI'].dropna().iloc[-1] if not df_sym['RSI'].dropna().empty else None
        st.write("RSI (14):", round(float(rsi_val),2) if rsi_val is not None else "نامشخص")
        if 'MACD_diff' in df_sym.columns:
            mac = df_sym['MACD_diff'].dropna().iloc[-1] if not df_sym['MACD_diff'].dropna().empty else None
            st.write("MACD diff:", round(float(mac),6) if mac is not None else "نامشخص")

    # raw table + download
    st.markdown("---")
    st.subheader("دادهٔ خام (آخرین ردیف‌ها)")
    st.dataframe(df_sym.tail(200))
    csv = df_sym.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ دانلود CSV", csv, file_name=f"{chosen}_data_{datetime.now().strftime('%Y%m%d')}.csv", mime='text/csv')

st.caption("⚠️ این ابزار آموزشی است؛ تصمیم‌گیری سرمایه‌گذاری نیست. همیشه ریسک را مدیریت کنید.")
