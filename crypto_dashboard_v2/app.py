import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from helpers import fetch_data, compute_indicators, predict_prophet, safe_get_news

# صفحه
st.set_page_config(page_title="Advanced Crypto Dashboard", layout="wide")
st.title("📈 Advanced Crypto Dashboard")

# Sidebar - تنظیمات
st.sidebar.header("Settings")
available = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Cardano (ADA)": "ADA-USD",
    "Solana (SOL)": "SOL-USD",
    "Ripple (XRP)": "XRP-USD",
    "Dogecoin (DOGE)": "DOGE-USD",
    "Polkadot (DOT)": "DOT-USD"
}

selected = st.sidebar.multiselect("Select cryptocurrencies (max 5):", list(available.keys()), default=["Bitcoin (BTC)"])
selected = selected[:5]  # حداکثر 5 تا برای نمایش همزمان

period_days = st.sidebar.selectbox("History range:", ["7 days", "30 days", "90 days", "180 days", "365 days"])
period_map = {"7 days":7, "30 days":30, "90 days":90, "180 days":180, "365 days":365}
days = period_map[period_days]

predict_days = st.sidebar.selectbox("Prediction horizon:", [3, 7, 14], index=0)

show_rsi = st.sidebar.checkbox("Show RSI", value=True)
show_ma = st.sidebar.checkbox("Show Moving Averages (MA20, MA50)", value=True)
show_news = st.sidebar.checkbox("Show News (may require API key)", value=False)

# Main layout: دو ستون بالا برای نمودارها/متریک‌ها، بخش پایین برای جزئیات
col1, col2 = st.columns([2,1])

# متریک: آخرین قیمت‌ها
with col2:
    st.subheader("Latest Prices")
    metrics = []
    for name in selected:
        symbol = available[name]
        df = fetch_data(symbol, days)
        if df is None or df.empty:
            st.metric(label=name, value="N/A")
        else:
            last = df["Close"].ffill().iloc[-1]
            change24 = None
            if len(df) >= 2:
                prev = df["Close"].ffill().iloc[-2]
                change24 = (last - prev) / prev * 100 if prev != 0 else 0.0
            if change24 is None:
                st.metric(label=name, value=f"{last:.4f}")
            else:
                st.metric(label=name, value=f"{last:.4f}", delta=f"{change24:.2f}%")

# نمودار قیمتی چند ارز در یک نمودار (در col1)
with col1:
    st.subheader("Price Chart")
    combined_fig = go.Figure()
    any_data = False
    for name in selected:
        symbol = available[name]
        df = fetch_data(symbol, days)
        if df is None or df.empty:
            continue
        any_data = True
        combined_fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=name))
    if any_data:
        combined_fig.update_layout(template="plotly_dark", height=420, legend=dict(orientation="h"))
        st.plotly_chart(combined_fig, use_container_width=True)
    else:
        st.warning("No data available for selected symbols / range.")

# جزئیات برای هر ارز بصورت تب
st.subheader("Detailed View")
for name in selected:
    symbol = available[name]
    st.markdown(f"---\n### {name} — `{symbol}`")
    df = fetch_data(symbol, days)
    if df is None or df.empty:
        st.error("No data available.")
        continue

    # محاسبه اندیکاتورها
    df2 = compute_indicators(df.copy(), ma_periods=(20,50))
    # جدول خلاصه
    c1, c2, c3 = st.columns(3)
    last_close = df2["Close"].ffill().iloc[-1]
    c1.metric("Last Close", f"{last_close:.6f}")
    if "MA20" in df2.columns and pd.notna(df2["MA20"].iloc[-1]):
        c2.metric("MA20", f"{df2['MA20'].iloc[-1]:.6f}")
    if "MA50" in df2.columns and pd.notna(df2["MA50"].iloc[-1]):
        c3.metric("MA50", f"{df2['MA50'].iloc[-1]:.6f}")

    # نمودار با MA و RSI (در صورت فعال بودن)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df2.index, y=df2["Close"], name="Close", line=dict(color="#FFD700")))
    if show_ma and "MA20" in df2.columns:
        fig.add_trace(go.Scatter(x=df2.index, y=df2["MA20"], name="MA20", line=dict(dash="dash")))
    if show_ma and "MA50" in df2.columns:
        fig.add_trace(go.Scatter(x=df2.index, y=df2["MA50"], name="MA50", line=dict(dash="dot")))
    fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig, use_container_width=True)

    if show_rsi and "RSI" in df2.columns:
        st.line_chart(df2["RSI"].dropna())

    # پیش‌بینی Prophet (اگر نصب شده باشد)
    with st.expander("🔮 Price prediction (Prophet)"):
        pred = predict_prophet(df2, days_ahead=predict_days)
        if isinstance(pred, str):
            st.info(pred)
        else:
            # pred expected columns: ds, yhat
            try:
                figp = px.line(pred, x="ds", y="yhat", title=f"Prophet forecast next {predict_days} days")
                st.plotly_chart(figp, use_container_width=True)
            except Exception as e:
                st.error("Error rendering prediction chart.")

    # اخبار (در صورت فعال بودن)
    if show_news:
        st.subheader("Related News")
        news_list = safe_get_news(name)
        if isinstance(news_list, str):
            st.info(news_list)
        elif len(news_list) == 0:
            st.info("No news found or API key not provided.")
        else:
            for n in news_list[:5]:
                st.write(f"- [{n.get('title')}]({n.get('url')}) — {n.get('source', '')}")

st.markdown("---\nBuilt with ❤️ — Advanced Crypto Dashboard")
