import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.graph_objects as go

# ===== تنظیمات صفحه =====
st.set_page_config(page_title="Crypto Dashboard", page_icon="💹", layout="wide")
st.title("💹 داشبورد تحلیل و پیش‌بینی ارز دیجیتال")
st.markdown("این داشبورد قیمت لحظه‌ای، سیگنال خرید/فروش و پیش‌بینی قیمت فردا را نمایش می‌دهد.")

# ===== لیست ارزها =====
symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "XRP-USD", "XLM-USD"]
names = {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "ADA-USD": "Cardano", "XRP-USD": "XRP", "XLM-USD": "Stellar"}

# ===== تابع دریافت داده =====
@st.cache_data
def get_data(symbol, period="7d", interval="1h"):
    df = yf.download(symbol, period=period, interval=interval)
    df.dropna(inplace=True)
    return df

# ===== تابع سیگنال =====
def get_signal(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    if df['MA5'].iloc[-1] > df['MA20'].iloc[-1]:
        return "📈 خرید", "green"
    elif df['MA5'].iloc[-1] < df['MA20'].iloc[-1]:
        return "📉 فروش", "red"
    else:
        return "⏸ نگه‌داری", "gray"

# ===== تابع پیش‌بینی =====
def predict_price(df):
    df['Return'] = df['Close'].pct_change()
    mean_return = df['Return'].mean()
    last_price = df['Close'].iloc[-1]
    return round(last_price * (1 + mean_return), 2)

# ===== نمایش جدول =====
table_data = []
for sym in symbols:
    df = get_data(sym)
    last_price = round(df['Close'].iloc[-1], 2)
    signal, color = get_signal(df)
    prediction = predict_price(df)
    table_data.append({
        "ارز": names[sym],
        "قیمت فعلی (USD)": last_price,
        "سیگنال": f"<span style='color:{color}; font-weight:bold'>{signal}</span>",
        "پیش‌بینی فردا (USD)": prediction
    })

df_table = pd.DataFrame(table_data)
st.markdown(df_table.to_html(escape=False, index=False), unsafe_allow_html=True)

# ===== انتخاب ارز برای نمودار =====
selected_symbol = st.selectbox("انتخاب ارز برای مشاهده نمودار", symbols, format_func=lambda x: names[x])
df_chart = get_data(selected_symbol)

# ===== نمودار =====
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'], mode='lines', name='قیمت', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'].rolling(5).mean(), mode='lines', name='MA5', line=dict(color='green')))
fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'].rolling(20).mean(), mode='lines', name='MA20', line=dict(color='red')))
fig.update_layout(title=f"نمودار قیمت {names[selected_symbol]}", xaxis_title="تاریخ", yaxis_title="USD", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)
