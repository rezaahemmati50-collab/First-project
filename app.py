# app.py — Enhanced AureumAI forecasting demo
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime, date

st.set_page_config(page_title="AureumAI Forecast (Demo)", layout="centered")
st.title("AureumAI — Crypto Forecast (Demo)")
st.markdown("پیش‌بینی امن و تشخیصی با Prophet — شامل بازهٔ عدم قطعیت. (USD)")

# ---------------------
# Helpers
# ---------------------
@st.cache_data(ttl=300)
def fetch_yahoo(symbol, period="3mo", interval="1d"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        return df
    except Exception as e:
        st.error(f"خطا در دریافت داده از Yahoo Finance: {e}")
        return pd.DataFrame()

def prepare_df(data):
    # expects DataFrame with DatetimeIndex and 'Close' column
    if data is None or data.empty:
        raise ValueError("داده خالی است.")
    df = data.copy()
    # ensure datetime index
    if not np.issubdtype(df.index.dtype, np.datetime64):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass
    df = df.reset_index()
    # find date col name
    date_col = df.columns[0]
    if 'Close' not in df.columns:
        raise ValueError("ستون 'Close' در دیتا وجود ندارد.")
    df = df[[date_col, 'Close']].rename(columns={date_col: 'ds', 'Close': 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['y','ds']).reset_index(drop=True)
    # ensure 1-d
    df = df[['ds','y']]
    return df

def fit_prophet(df, changepoint_prior_scale=0.05):
    # choose whether to log-transform
    use_log = (df['y'] > 0).all()
    df_fit = df.copy()
    if use_log:
        df_fit['y'] = np.log(df_fit['y'])
    m = Prophet(daily_seasonality=False, weekly_seasonality=True,
                yearly_seasonality=False, changepoint_prior_scale=changepoint_prior_scale,
                seasonality_mode='multiplicative')
    m.fit(df_fit.rename(columns={'y':'y'}))
    return m, use_log

def predict_days(m, use_log, periods):
    future = m.make_future_dataframe(periods=periods, freq='D')
    forecast = m.predict(future)
    # revert log if used
    if use_log:
        forecast['yhat_final'] = np.exp(forecast['yhat'])
        forecast['yhat_lower_final'] = np.exp(forecast['yhat_lower'])
        forecast['yhat_upper_final'] = np.exp(forecast['yhat_upper'])
    else:
        forecast['yhat_final'] = forecast['yhat']
        forecast['yhat_lower_final'] = forecast['yhat_lower']
        forecast['yhat_upper_final'] = forecast['yhat_upper']
    return forecast

# ---------------------
# UI controls
# ---------------------
assets = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "Stellar (XLM)": "XLM-USD",
    "Solana (SOL)": "SOL-USD",
    "Litecoin (LTC)": "LTC-USD",
    "Ripple (XRP)": "XRP-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Polkadot (DOT)": "DOT-USD"
}

col1, col2 = st.columns([2,1])
with col1:
    asset_name = st.selectbox("انتخاب ارز دیجیتال", list(assets.keys()), index=1)
    symbol = assets[asset_name]
with col2:
    period = st.selectbox("دورهٔ داده", ["1mo","3mo","6mo","1y"], index=1)
days = st.selectbox("تعداد روز پیش‌بینی", [1,3,7,30], index=1)
changepoint_prior_scale = st.slider("حساسیت مدل به تغییر روند (changepoint_prior_scale)", 0.001, 0.5, 0.05, step=0.01)

st.markdown("---")

# option to upload CSV instead of yfinance
uploaded = st.file_uploader("آپلود CSV (اختیاری) — ستون Date و Close", type=['csv'])
if uploaded:
    try:
        df_csv = pd.read_csv(uploaded)
        # try auto-detect date column
        if 'Date' in df_csv.columns:
            df_csv['Date'] = pd.to_datetime(df_csv['Date'])
            df_csv = df_csv.set_index('Date')
        else:
            # if first column is date-like
            df_csv.iloc[:,0] = pd.to_datetime(df_csv.iloc[:,0])
            df_csv = df_csv.set_index(df_csv.columns[0])
        data_raw = df_csv
        st.success("CSV دریافت شد و به عنوان منبع داده انتخاب شد.")
    except Exception as e:
        st.error(f"خطا در خواندن CSV: {e}")
        st.stop()
else:
    with st.spinner("در حال دانلود داده از Yahoo Finance ..."):
        data_raw = fetch_yahoo(symbol, period=period, interval="1d")

# check emptiness
if data_raw is None or data_raw.empty:
    st.error("داده‌ای در دسترس نیست. اتصال اینترنت را بررسی کن یا فایل CSV آپلود کن.")
    st.stop()

# show tail raw
st.subheader("دادهٔ خام (آخرین 10 سطر)")
try:
    st.dataframe(data_raw[['Close']].tail(10))
except Exception:
    st.write(data_raw.tail(10))

# currency hint
if symbol.endswith("-USD"):
    st.info("توجه: قیمت‌ها بر حسب USD هستند.")
elif symbol.endswith("-CAD"):
    st.info("توجه: قیمت‌ها بر حسب CAD هستند.")

# prepare for prophet
try:
    df_prepared = prepare_df(data_raw)
except Exception as e:
    st.error(f"خطا در آماده‌سازی داده: {e}")
    st.stop()

st.subheader("آماده‌سازی دیتا برای مدل")
st.write(df_prepared.tail(5))

if len(df_prepared) < 10:
    st.warning("دقت کن: طول داده کمتر از 10 سطر است — پیش‌بینی قابل اعتماد نخواهد بود.")

# fit model
with st.spinner("در حال فیت مدل Prophet ..."):
    try:
        model, used_log = fit_prophet(df_prepared, changepoint_prior_scale=changepoint_prior_scale)
    except Exception as e:
        st.error(f"خطا در فیت مدل: {e}")
        st.stop()

# predict
with st.spinner("در حال پیش‌بینی ..."):
    forecast = predict_days(model, used_log, periods=days)

# extract results for next 'days' days
predicted = forecast[['ds','yhat_final','yhat_lower_final','yhat_upper_final']].tail(days).reset_index(drop=True)
predicted.columns = ['ds','yhat','yhat_lower','yhat_upper']
predicted['ds'] = pd.to_datetime(predicted['ds']).dt.date
predicted[['yhat','yhat_lower','yhat_upper']] = predicted[['yhat','yhat_lower','yhat_upper']].round(2)

st.subheader(f"پیش‌بینی {days} روز آینده")
st.table(predicted.rename(columns={'ds':'تاریخ','yhat':'پیش‌بینی','yhat_lower':'حد پایین','yhat_upper':'حد بالا'}))

# show last close vs first predicted
last_close = df_prepared['y'].iloc[-1]
st.metric("آخرین Close (داده شده)", f"{last_close:.2f}")

# plot
st.subheader("نمودار: قیمت واقعی و پیش‌بینی (خط طلا = yhat)")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df_prepared['ds'], df_prepared['y'], label='Actual', color='white')
ax.plot(forecast['ds'], forecast['yhat_final'], label='Forecast', color='#FFD700')  # gold
ax.fill_between(forecast['ds'], forecast['yhat_lower_final'], forecast['yhat_upper_final'], color='gray', alpha=0.2)
ax.set_facecolor('#0e1117')
fig.patch.set_facecolor('#0e1117')
ax.tick_params(colors='white', which='both')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.markdown("**نکات:**")
st.markdown("""
- عدد پیش‌بینی (yhat) نقطهٔ انتظاری است؛ حتماً بازهٔ عدم قطعیت (yhat_lower — yhat_upper) را بررسی کنید.  
- اگر yhat کمتر از قیمت فعلی است اما داخل بازهٔ عدم قطعیت قرار دارد، مدل هنوز می‌تواند با احتمال معقولی قیمت فعلی را پوشش دهد.  
- برای حساسیت بیشتر به نوسانات روزانه، مقدار changepoint_prior_scale را افزایش دهید (مثلاً 0.2 یا 0.3)؛ اگر نویز میبینید آن را کاهش دهید.
""")

st.caption("نسخهٔ اصلاح‌شدهٔ AureumAI demo — برای ارتقا: می‌توان مدل دوم (ARIMA/LSTM) اضافه کرده و از اجماع مدل‌ها (ensemble) استفاده کرد.")
