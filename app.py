import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

st.set_page_config(page_title="Crypto Price Prediction", layout="wide")
st.title("📈 Cryptocurrency Price Prediction App")

# Sidebar inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter cryptocurrency ticker (e.g., BTC-USD):", "BTC-USD")
n_years = st.sidebar.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Load data
@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"❌ Failed to load data: {e}")
        return pd.DataFrame()

df = load_data(ticker)

# Check if data exists
if df.empty or 'Close' not in df.columns:
    st.error("❌ No data found for the given ticker. Please try another one.")
    st.stop()

# Show raw data
st.subheader('Raw Data')
st.write(df.tail())

# Plot raw data
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close Price'))
fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Prepare data for Prophet
df_train = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Ensure numeric values for y
if not pd.api.types.is_numeric_dtype(df_train['y']):
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')

df_train = df_train.dropna()

# Stop if no valid data
if df_train.empty:
    st.error("❌ No valid numeric 'Close' price data available for forecasting.")
    st.stop()

# Train Prophet model
m = Prophet()
m.fit(df_train)

# Future dataframe
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show forecast
st.subheader('Forecast Data')
st.write(forecast.tail())

# Plot forecast
st.subheader('Forecast Plot')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# Forecast components
st.subheader("Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)
