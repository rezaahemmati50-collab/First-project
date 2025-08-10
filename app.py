# app.py
# داشبورد کامل تحلیل و پیش‌بینی کریپتو — فارسی
# ویژگی‌ها:
# - انتخاب ارز و بازه (شامل بازه‌های کوتاه 1d/5d/1wk/2wk)
# - دانلود داده با interval مناسب
# - محاسبه RSI و MACD (در صورت نبودن پکیج ta از محاسبات داخلی استفاده می‌شود)
# - پیش‌بینی 3 و 7 روزه با Prophet در صورت نصب؛ در غیر این صورت fallback سریع
# - تولید سیگنال خرید/فروش/نگهداری با دلیل
# - نمودار قیمت، MAها و پیش‌بینی‌ها با Plotly
# - مدیریت کامل خطاها و داده‌های ناقص

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Crypto Analyst · پیش‌بینی و سیگنال", layout="wide")

# ----- try optional libraries -----
HAS_PROPHET = False
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly
    HAS_PROPHET = True
except Exception:
    HAS_PROPHET = False

HAS_TA = False
try:
    import ta
    HAS_TA = True
except Exception:
    HAS_TA = False

# ----- Helpers -----
def simple_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Fallback محاسبه RSI اگر پکیج ta موجود نباشد."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-10)
    return 100 - (100 / (1 + rs))

def simple_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Fallback محاسبه MACD ساده (macd, macd_signal, macd_diff)"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_diff = macd - macd_signal
    return macd, macd_signal, macd_diff

def ensure_flat_columns(df: pd.DataFrame) -> pd.DataFrame:
    """اگر MultiIndex برگشت داده شود، ستون‌ها را صاف می‌کند."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df

def fetch_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    """دریافت ایمن داده از yfinance"""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df is None:
            return pd.DataFrame()
        df = ensure_flat_columns(df)
        return df
    except Exception as e:
        return pd.DataFrame()

def moving_avg_forecast(series: pd.Series, days: int):
    """Fallback سریع: رشد میانگین درصدی ساده"""
    if series.dropna().empty:
        return np.array([np.nan]*days)
    last = float(series.dropna().iloc[-1])
    avg_pct = series.pct_change().dropna().mean()
    if np.isnan(avg_pct):
        avg_pct = 0.0
    return np.array([ last * ((1+avg_pct)**i) for i in range(1, days+1) ])

def compute_signal(df: pd.DataFrame, forecast_next: float = None):
    """
    ترکیب ساده از RSI و MACD و پیش‌بینی برای دادن سیگنال.
    خروجی: (label, color, reason_text)
    """
    reasons = []
    score = 0  # مثبت => خرید، منفی => فروش

    close = df['Close'].dropna()
    if close.empty:
        return ("⚪ داده ناکافی", "gray", "داده قیمت موجود نیست")

    # RSI
    try:
        if HAS_TA:
            rsi = ta.momentum.RSIIndicator(close, window=14).rsi()
            last_rsi = float(rsi.iloc[-1])
        else:
            last_rsi = float(simple_rsi(close, 14).iloc[-1])
        if last_rsi < 30:
            score += 1
            reasons.append(f"RSI پایین ({last_rsi:.1f}) → اشباع فروش")
        elif last_rsi > 70:
            score -= 1
            reasons.append(f"RSI بالا ({last_rsi:.1f}) → اشباع خرید")
        else:
            reasons.append(f"RSI نرمال ({last_rsi:.1f})")
    except Exception as e:
        reasons.append("RSI محاسبه نشد")

    # MACD
    try:
        if HAS_TA:
            macd_obj = ta.trend.MACD(close)
            macd_diff = float(macd_obj.macd_diff().iloc[-1])
        else:
            _, _, macd_diff = simple_macd(close)
            macd_diff = float(macd_diff.iloc[-1])
        if macd_diff > 0:
            score += 1
            reasons.append(f"MACD مثبت ({macd_diff:.4f}) → روند صعودی")
        elif macd_diff < 0:
            score -= 1
            reasons.append(f"MACD منفی ({macd_diff:.4f}) → روند نزولی")
        else:
            reasons.append("MACD خنثی")
    except Exception as e:
        reasons.append("MACD محاسبه نشد")

    # MA cross (MA20/MA50)
    try:
        ma20 = df['MA20'].dropna()
        ma50 = df['MA50'].dropna()
        if not ma20.empty and not ma50.empty:
            last_ma20 = float(ma20.iloc[-1]); last_ma50 = float(ma50.iloc[-1])
            if last_ma20 > last_ma50:
                score += 1
                reasons.append("MA20 بالاتر از MA50 → سیگنال صعودی")
            elif last_ma20 < last_ma50:
                score -= 1
                reasons.append("MA20 پایین‌تر از MA50 → سیگنال نزولی")
            else:
                reasons.append("MA کراس خنثی")
        else:
            reasons.append("MA20/MA50 ناکافی")
    except Exception:
        reasons.append("خطا در محاسبه MA")

    # Forecast influence
    if forecast_next is not None and not np.isnan(forecast_next):
        last_price = float(close.iloc[-1])
        pct = (forecast_next - last_price) / last_price if last_price != 0 else 0
        if pct > 0.01:
            score += 1
            reasons.append(f"پیش‌بینی +{pct*100:.2f}% → حمایت کوتاه‌مدت")
        elif pct < -0.01:
            score -= 1
            reasons.append(f"پیش‌بینی {pct*100:.2f}% → ریسک کوتاه‌مدت")
        else:
            reasons.append(f"پیش‌بینی تغییر جزئی ({pct*100:.2f}%)")

    # Decide
    if score >= 3:
        return ("🔵 سیگنال قوی خرید (Strong BUY)", "green", " · ".join(reasons))
    elif score == 2:
        return ("🟢 سیگنال خرید (BUY)", "green", " · ".join(reasons))
    elif score == 1:
        return ("🟡 تمایل به خرید (Weak BUY)", "goldenrod", " · ".join(reasons))
    elif score == 0:
        return ("⚪ نگه‌داری (HOLD)", "gray", " · ".join(reasons))
    elif score == -1:
        return ("🟠 سیگنال فروش (SELL)", "orangered", " · ".join(reasons))
    elif score <= -2:
        return ("🔴 سیگنال قوی فروش (Strong SELL)", "red", " · ".join(reasons))
    else:
        return ("⚪ نگه‌داری (HOLD)", "gray", " · ".join(reasons))


# ----- UI: Controls -----
st.markdown("<h1 style='text-align:center;'>AureumAI · Crypto Analyzer</h1>", unsafe_allow_html=True)
col_a, col_b, col_c = st.columns([2,1,1])
with col_a:
    asset = st.selectbox("انتخاب ارز (پیش‌فرض BTC):", ["BTC-USD","ETH-USD","ADA-USD","XLM-USD"], index=0)
with col_b:
    period_choice = st.selectbox("بازه زمانی:", ["1d","5d","1wk","2wk","1mo","3mo","6mo","1y"], index=1)
with col_c:
    chart_interval = None
    # interval انتخاب خودکار بر اساس period_choice
    if period_choice in ["1d","5d"]:
        chart_interval = st.selectbox("دقت داده (interval):", ["1m","5m","15m"], index=1)
    else:
        chart_interval = st.selectbox("دقت داده (interval):", ["1d","1h"], index=0)

# forecast days fixed: show both 3 and 7
forecast_days_list = [3,7]

st.markdown("---")

# ----- Fetch data -----
with st.spinner("دریافت داده‌ها از Yahoo Finance ..."):
    data = fetch_data(asset, period_choice, chart_interval)

if data.empty:
    st.error("❌ داده‌ای برای این انتخاب یافت نشد — لطفاً بازه یا نماد دیگری انتخاب کنید.")
    st.stop()

# ensure flat columns and required columns exist
data = ensure_flat_columns(data)
if "Close" not in data.columns:
    # try variations
    possible = [c for c in data.columns if "Close" in str(c)]
    if possible:
        data.rename(columns={possible[0]:"Close"}, inplace=True)

if "Close" not in data.columns or data["Close"].dropna().empty:
    st.error("❌ ستون Close یا دادهٔ قیمت معتبر یافت نشد.")
    st.stop()

# ----- Feature engineering -----
# make sure Date column exists for prophet compatibility
data = data.copy()
if np.issubdtype(data.index.dtype, np.datetime64):
    data.reset_index(inplace=True)
    # keep index name as Date for later
    if 'Datetime' in data.columns:
        data.rename(columns={'Datetime':'Date'}, inplace=True)
# normalize Date column name (some intervals give 'Datetime' column)
if 'Date' not in data.columns and 'datetime' in [c.lower() for c in data.columns]:
    # find the column
    for c in data.columns:
        if c.lower() == 'datetime':
            data.rename(columns={c:'Date'}, inplace=True)
            break

if 'Date' not in data.columns:
    # if we couldn't find a 'Date' column but index is datetime, set it
    if np.issubdtype(data.index.dtype, np.datetime64):
        data = data.reset_index().rename(columns={data.columns[0]:'Date'})

# ensure 'Date' and 'Close' exist
if 'Date' not in data.columns:
    # if still not, create Date from index
    data.reset_index(inplace=True)
    if data.columns[0] != 'Date':
        data = data.rename(columns={data.columns[0]:'Date'})

# convert types
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Date','Close']).sort_values('Date').reset_index(drop=True)

# calculate moving averages (on Close)
data['MA7'] = data['Close'].rolling(window=7, min_periods=1).mean()
data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()

# indicators
try:
    if HAS_TA:
        data['RSI'] = ta.momentum.RSIIndicator(data['Close'], window=14).rsi()
        macd_obj = ta.trend.MACD(data['Close'])
        data['MACD'] = macd_obj.macd()
        data['MACD_signal'] = macd_obj.macd_signal()
        data['MACD_diff'] = macd_obj.macd_diff()
    else:
        data['RSI'] = simple_rsi(data['Close'], 14)
        macd, macd_signal, macd_diff = simple_macd(data['Close'])
        data['MACD'] = macd
        data['MACD_signal'] = macd_signal
        data['MACD_diff'] = macd_diff
except Exception:
    data['RSI'] = simple_rsi(data['Close'], 14)
    macd, macd_signal, macd_diff = simple_macd(data['Close'])
    data['MACD'] = macd
    data['MACD_signal'] = macd_signal
    data['MACD_diff'] = macd_diff

# ----- Headline metrics -----
col1, col2, col3, col4 = st.columns([2,1,1,1])
col1.metric("نماد", asset)
last_close = float(data['Close'].iloc[-1])
change_24h = None
# compute approximate 24h change if possible (find value 24h ago)
try:
    # find nearest timestamp ~24h ago
    ts_target = data['Date'].iloc[-1] - pd.Timedelta(days=1)
    prev_row = data[data['Date'] <= ts_target]
    if not prev_row.empty:
        prev_val = float(prev_row['Close'].iloc[-1])
        change_24h = (last_close - prev_val) / prev_val * 100
except Exception:
    change_24h = None

col2.metric("قیمت آخرین", f"${last_close:,.2f}")
col3.metric("تغییر ~24 ساعت", f"{change_24h:.2f}%" if change_24h is not None else "نامشخص")
col4.metric("حجم (آخرین)", f"{int(data['Volume'].iloc[-1])}" if 'Volume' in data.columns else "نامشخص")

st.markdown("---")

# ----- Price chart (history) -----
st.subheader("📈 نمودار قیمت و میانگین‌ها")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close', line=dict(color='#00BFA6')))
fig.add_trace(go.Scatter(x=data['Date'], y=data['MA7'], mode='lines', name='MA7', line=dict(color='#FFD54F')))
fig.add_trace(go.Scatter(x=data['Date'], y=data['MA20'], mode='lines', name='MA20', line=dict(color='#FF7043')))
if 'MA50' in data.columns:
    fig.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], mode='lines', name='MA50', line=dict(color='#B39DDB')))

fig.update_layout(template='plotly_dark', height=420, margin=dict(t=30,b=10))
st.plotly_chart(fig, use_container_width=True)

# ----- Indicators panel -----
with st.expander("📊 اندیکاتورها (RSI و MACD)"):
    st.write("RSI (14):", round(float(data['RSI'].dropna().iloc[-1]),2) if not data['RSI'].dropna().empty else "نامشخص")
    st.write("MACD diff (آخرین):", round(float(data['MACD_diff'].dropna().iloc[-1]),6) if not data['MACD_diff'].dropna().empty else "نامشخص")
    st.line_chart(data.set_index('Date')[['RSI']].dropna())

st.markdown("---")

# ----- Forecast (Prophet or fallback) -----
st.subheader("🔮 پیش‌بینی کوتاه‌مدت (۳ و ۷ روز)")
forecast_vals_3 = None
forecast_vals_7 = None
used_prophet = False

# prepare df for prophet (daily expected)
prophet_df = data[['Date','Close']].rename(columns={'Date':'ds','Close':'y'}).copy()
# Prophet requires at least 2 rows
if prophet_df['y'].dropna().shape[0] >= 2 and HAS_PROPHET:
    try:
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=7, freq='D')
        pred = m.predict(future)
        # take next 3 and 7 days
        last_pred = pred.tail(7)
        forecast_vals_3 = last_pred['yhat'].head(3).values if len(last_pred)>=3 else last_pred['yhat'].values
        forecast_vals_7 = last_pred['yhat'].values
        used_prophet = True
    except Exception:
        # fallback to simple method
        forecast_vals_7 = moving_avg_forecast(data['Close'], 7)
        forecast_vals_3 = forecast_vals_7[:3]
        used_prophet = False
else:
    # quick fallback (moving avg pct)
    forecast_vals_7 = moving_avg_forecast(data['Close'], 7)
    forecast_vals_3 = forecast_vals_7[:3]
    used_prophet = False

# display numeric forecast
colf1, colf2 = st.columns(2)
with colf1:
    if len(forecast_vals_3)>0:
        st.metric("پیش‌بینی ۳ روزه (اولین)", f"${float(forecast_vals_3[0]):,.2f}")
    else:
        st.write("پیش‌بینی ۳ روزه: نامشخص")
with colf2:
    if len(forecast_vals_7)>0:
        st.metric("پیش‌بینی ۷ روزه (روز اول)", f"${float(forecast_vals_7[0]):,.2f}")
    else:
        st.write("پیش‌بینی ۷ روزه: نامشخص")

# Forecast combined chart (history + 3/7)
st.markdown("### نمودار ترکیبی: قیمت واقعی و پیش‌بینی")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Actual', line=dict(color='#9AA0A6')))

# build forecast date axis starting from next day
start_date = data['Date'].iloc[-1] + pd.Timedelta(days=1)
dates_7 = [start_date + pd.Timedelta(days=i) for i in range(len(forecast_vals_7))]
if len(forecast_vals_3)>0:
    dates_3 = dates_7[:len(forecast_vals_3)]
else:
    dates_3 = []

if len(forecast_vals_7)>0:
    fig2.add_trace(go.Scatter(x=dates_7, y=forecast_vals_7, mode='lines+markers', name='Forecast 7d', line=dict(color='#FFA500', dash='dash')))
if len(forecast_vals_3)>0:
    fig2.add_trace(go.Scatter(x=dates_3, y=forecast_vals_3, mode='lines+markers', name='Forecast 3d', line=dict(color='#00B0FF', dash='dot')))

fig2.update_layout(template='plotly_dark', height=420)
st.plotly_chart(fig2, use_container_width=True)

# ----- Compute signal using forecast first predicted day -----
next_forecast = float(forecast_vals_3[0]) if (forecast_vals_3 is not None and len(forecast_vals_3)>0 and not np.isnan(forecast_vals_3[0])) else None
signal_label, signal_color, signal_reason = compute_signal(data, forecast_next=next_forecast)

st.markdown("---")
st.markdown(f"<div style='padding:16px; border-radius:8px; background-color:#0f1720;'><h2 style='color:{signal_color}; margin:0;'>{signal_label}</h2><p style='color:#bdbdbd; margin:6px 0 0 0;'>{signal_reason}</p></div>", unsafe_allow_html=True)

# ----- Data table and download -----
st.subheader("📄 دادهٔ قیمتی (آخرین ردیف‌ها)")
st.dataframe(data.tail(30))

csv = data.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ دانلود CSV داده‌ها", csv, file_name=f"{asset}_data_{datetime.now().strftime('%Y%m%d')}.csv", mime='text/csv')

st.caption("تذکر: این ابزار تحلیلی آموزشی است و توصیهٔ سرمایه‌گذاری محسوب نمی‌شود. همیشه ریسک را مدیریت کنید.")
