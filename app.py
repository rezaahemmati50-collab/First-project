# app.py
# داشبورد ساده و پایدار سیگنال‌گیری با Moving Average Crossover (MA20 / MA50)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="MA Crossover Signal · Crypto", layout="wide")
st.title("📈 ابزار سیگنال میانگین متحرک (MA20 / MA50) - ساده و مقاوم")

# ---------------------------
# تنظیمات UI
# ---------------------------
col1, col2 = st.columns([2,1])
with col1:
    symbol = st.selectbox("انتخاب نماد (نمونه):", ["BTC-USD","ETH-USD","ADA-USD","XLM-USD"], index=0)
with col2:
    period_choice = st.selectbox("بازه زمانی (period):", ["1d","5d","1wk","2wk","1mo","3mo","6mo","1y"], index=1)

# برای بازه‌های کوتاه، به کاربر اجازه می‌دهیم interval را انتخاب کند
if period_choice in ["1d","5d"]:
    interval = st.selectbox("دقت داده (interval):", ["1m","5m","15m","30m","60m"], index=1)
else:
    interval = st.selectbox("دقت داده (interval):", ["1d","1h"], index=0)

st.markdown("---")

# ---------------------------
# دریافت داده (safe)
# ---------------------------
@st.cache_data(ttl=180)
def fetch_safe(sym: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(sym, period=period, interval=interval, progress=False)
        if df is None:
            return pd.DataFrame()
        # اگر MultiIndex داشت، تختش کن
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

with st.spinner("دریافت داده از Yahoo Finance ..."):
    raw = fetch_safe(symbol, period_choice, interval)

# ---------------------------
# چک اولیه و نرمال‌سازی تاریخ
# ---------------------------
if raw.empty:
    st.error("⚠️ داده‌ای برای این نماد/بازه یافت نشد. لطفاً نماد یا بازه/دقت دیگری انتخاب کنید.")
    st.stop()

data = raw.copy()

# تلاش برای تبدیل index به datetime در صورت نیاز
try:
    if not np.issubdtype(data.index.dtype, np.datetime64):
        data.index = pd.to_datetime(data.index, errors='coerce')
except Exception:
    # fallback: اگر index نتوانست تبدیل شود، ریست کن به ستون
    data = data.reset_index()

# اگر ستون تاریخ به شکل Datetime در ستون‌هاست، آنرا به 'Date' نام‌گذاری کن
if 'Date' not in data.columns:
    # اگر اولین ستون نامش چیزی مثل 'Datetime' یا index است، ری‌نام کن
    for c in data.columns:
        if c.lower() in ('datetime','date','index'):
            data = data.rename(columns={c:'Date'})
            break

# اگر هنوز 'Date' در ستون‌ها نیست، ریست ایندکس و نام‌گذاری کن
if 'Date' not in data.columns:
    data = data.reset_index().rename(columns={data.columns[0]:'Date'})

# تبدیل و پاک‌سازی تاریخ و قیمت
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
# پیدا کن نزدیک‌ترین ستون قیمت: 'Close' یا variations
if 'Close' not in data.columns:
    close_candidates = [c for c in data.columns if 'close' in str(c).lower()]
    if close_candidates:
        data = data.rename(columns={close_candidates[0]:'Close'})

if 'Close' not in data.columns:
    st.error("❌ ستون 'Close' یافت نشد در داده دریافتی.")
    st.stop()

data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
# اگر Volume هم موجود است، عددی کن
if 'Volume' in data.columns:
    data['Volume'] = pd.to_numeric(data['Volume'], errors='coerce')

# حذف ردیف‌های مشکل‌دار و مرتب‌سازی
data = data.dropna(subset=['Date','Close']).sort_values('Date').reset_index(drop=True)

if data.empty:
    st.error("⚠️ پس از پاک‌سازی، داده‌ای باقی نماند.")
    st.stop()

# ---------------------------
# محاسبه MAها با min_periods
# ---------------------------
data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()

# ---------------------------
# نمایش متریک‌های اصلی
# ---------------------------
last_close = float(data['Close'].iloc[-1])
# محاسبه تغییر ~24h در صورت وجود
change_24h = None
try:
    t0 = data['Date'].iloc[-1] - pd.Timedelta(days=1)
    prev = data[data['Date'] <= t0]
    if not prev.empty:
        prev_close = float(prev['Close'].iloc[-1])
        change_24h = (last_close - prev_close) / prev_close * 100
except Exception:
    change_24h = None

c1, c2, c3 = st.columns([2,1,1])
c1.metric("نماد", symbol)
c2.metric("آخرین قیمت", f"${last_close:,.2f}")
c3.metric("~تغییر 24ساعته", f"{change_24h:.2f}%" if change_24h is not None else "نامشخص")

st.markdown("---")

# ---------------------------
# نمودار قیمت و MA (Plotly)
# ---------------------------
st.subheader("📈 نمودار قیمت و میانگین‌های متحرک")
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close', line=dict(color='#00BFA6')))
fig.add_trace(go.Scatter(x=data['Date'], y=data['MA20'], mode='lines', name='MA20', line=dict(color='#FFD54F')))
fig.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], mode='lines', name='MA50', line=dict(color='#FF7043')))

fig.update_layout(template='plotly_dark', height=420, margin=dict(t=30,b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# منطق سیگنال (MA crossover)
# ---------------------------
# وضعیت MAها: آیا MA50 داده‌ی واقعی (غیر-constant) دارد؟
ma50_valid = data['MA50'].dropna().shape[0] >= 2  # اگر بیش از 1 مقدار داشته باشیم اعتبار دارد
ma20_valid = data['MA20'].dropna().shape[0] >= 1

signal_text = "⚪ داده ناکافی"
signal_color = "gray"
signal_reason = ""

try:
    if ma50_valid:
        ma20_last = float(data['MA20'].dropna().iloc[-1])
        ma50_last = float(data['MA50'].dropna().iloc[-1])
        # سیگنال بر اساس کراس اوور
        if ma20_last > ma50_last:
            signal_text = "🔵 سیگنال خرید (MA20 > MA50)"
            signal_color = "green"
            signal_reason = f"MA20 ({ma20_last:.4f}) بالاتر از MA50 ({ma50_last:.4f})."
        elif ma20_last < ma50_last:
            signal_text = "🔴 سیگنال فروش (MA20 < MA50)"
            signal_color = "red"
            signal_reason = f"MA20 ({ma20_last:.4f}) پایین‌تر از MA50 ({ma50_last:.4f})."
        else:
            signal_text = "🟡 خنثی (MA20 == MA50)"
            signal_color = "goldenrod"
            signal_reason = "افق MAها خنثی است."
    elif ma20_valid:
        # فقط MA20 معتبر است → مقایسه با قیمت آخر
        ma20_last = float(data['MA20'].dropna().iloc[-1])
        if last_close > ma20_last:
            signal_text = "🟢 خرید (بر اساس MA20)"
            signal_color = "green"
            signal_reason = f"قیمت اخیر ${last_close:.2f} بالاتر از MA20 ({ma20_last:.4f})."
        elif last_close < ma20_last:
            signal_text = "🟠 فروش (بر اساس MA20)"
            signal_color = "orangered"
            signal_reason = f"قیمت اخیر ${last_close:.2f} پایین‌تر از MA20 ({ma20_last:.4f})."
        else:
            signal_text = "🟡 خنثی (بر اساس MA20)"
            signal_color = "goldenrod"
            signal_reason = "دادهٔ MA20 نشان‌دهندهٔ حالت خنثی است."
    else:
        signal_text = "⚪ داده کافی برای سیگنال‌دهی نیست"
        signal_color = "gray"
        signal_reason = "MA20 و MA50 دادهٔ کافی ندارند."
except Exception as e:
    signal_text = "⚪ خطا در محاسبه سیگنال"
    signal_color = "gray"
    signal_reason = str(e)

st.markdown("---")
st.markdown(f"<div style='padding:14px; border-radius:8px; background:#0b1220;'><h2 style='color:{signal_color}; margin:0;'>{signal_text}</h2><p style='color:#bdbdbd; margin:6px 0 0 0;'>{signal_reason}</p></div>", unsafe_allow_html=True)

# ---------------------------
# دیتا و دانلود
# ---------------------------
st.markdown("---")
st.subheader("📄 داده‌های قیمتی (آخرین ردیف‌ها)")
st.dataframe(data.tail(50))

csv = data.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ دانلود CSV داده‌ها", csv, file_name=f"{symbol}_data_{datetime.now().strftime('%Y%m%d')}.csv", mime='text/csv')

st.caption("تذکر: این ابزار آموزشی است و توصیهٔ سرمایه‌گذاری نیست. قبل از معامله، تحلیل و مدیریت ریسک انجام دهید.")
