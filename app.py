import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.title("📈 پیش‌بینی قیمت ارز دیجیتال (Prophet + LSTM)")

# انتخاب ارز و بازه
symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "XLM-USD"]
symbol = st.selectbox("ارز دیجیتال", symbols)
period = st.selectbox("بازه زمانی", ["1y", "6mo", "3mo", "1mo"])

# دانلود داده‌ها
@st.cache_data
def load_data(symbol, period):
    df = yf.download(symbol, period=period)
    df.reset_index(inplace=True)
    return df

df = load_data(symbol, period)
st.subheader("داده‌های اولیه")
st.dataframe(df.tail())

# Prophet Model
st.subheader("🔮 پیش‌بینی با Prophet")
df_train = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

# حذف داده‌های NaN یا غیرعددی
df_train = df_train.dropna()
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train = df_train.dropna()

if df_train.empty:
    st.error("❌ داده‌ای برای آموزش Prophet پیدا نشد. لطفاً بازه یا ارز را تغییر دهید.")
else:
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    fig1 = m.plot(forecast)
    st.pyplot(fig1)

# LSTM Model
st.subheader("🤖 پیش‌بینی با LSTM")
if len(df) < 60:
    st.warning("داده برای آموزش LSTM کافی نیست.")
else:
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=1, batch_size=1, verbose=0)

    predicted_price = model.predict(X)
    predicted_price = scaler.inverse_transform(predicted_price)

    # رسم نمودار
    fig2, ax = plt.subplots()
    ax.plot(df['Date'][60:], data[60:], label="واقعی")
    ax.plot(df['Date'][60:], predicted_price, label="پیش‌بینی")
    ax.legend()
    st.pyplot(fig2)
