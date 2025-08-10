import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import feedparser
import datetime as dt

# =========================
# تنظیمات اولیه صفحه
# =========================
st.set_page_config(
    page_title="Crypto Market Dashboard",
    layout="wide",
    page_icon="💹"
)

st.title("💹 Crypto Market Dashboard")
st.markdown("### پیش‌بینی قیمت، تحلیل و اخبار ارزهای دیجیتال")

# =========================
# لیست ارزها
# =========================
crypto_symbols = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Cardano": "ADA-USD",
    "XRP": "XRP-USD",
    "Stellar": "XLM-USD"
}

# انتخاب ارز
selected_crypto = st.selectbox("یک ارز دیجیتال انتخاب کنید:", list(crypto_symbols.keys()))
symbol = crypto_symbols[selected_crypto]

# =========================
# دریافت داده‌های بازار
# =========================
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, period="1y", interval="1d")
    data.reset_index(inplace=True)
    return data

data = load_data(symbol)

# =========================
# نمایش نمودار قیمت
# =========================
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="قیمت پایانی"))
fig.layout.update(title=f"نمودار قیمت {selected_crypto}", xaxis_rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=True)

# =========================
# پیش‌بینی با Prophet
# =========================
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

st.subheader("📈 پیش‌بینی 30 روز آینده")
fig2 = plot_plotly(model, forecast)
st.plotly_chart(fig2, use_container_width=True)

# =========================
# ستون اخبار (RSS)
# =========================
st.subheader("📰 آخرین اخبار بازار ارز دیجیتال")
rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
feed = feedparser.parse(rss_url)

if len(feed.entries) > 0:
    for entry in feed.entries[:5]:
        st.markdown(f"**[{entry.title}]({entry.link})**")
        st.caption(entry.published)
else:
    st.write("هیچ خبری یافت نشد.")

# =========================
# جدول داده‌ها
# =========================
st.subheader("📊 داده‌های خام بازار")
st.dataframe(data.tail(10))
