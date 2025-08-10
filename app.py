import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# App title
st.title("📈 Cryptocurrency Price Forecast App")

# Sidebar inputs
st.sidebar.header("Settings")
selected_ticker = st.sidebar.text_input("Enter cryptocurrency ticker (e.g., BTC-USD)", "BTC-USD")
n_years = st.sidebar.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Load data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="5y")
    df.reset_index(inplace=True)
    return df

data_load_state = st.text("Loading data...")
df = load_data(selected_ticker)
data_load_state.text("✅ Data loaded successfully!")

# Display raw data
st.subheader("Raw Data")
st.write(df.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close Price"))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Prepare data for Prophet
df_train = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

# Ensure y is numeric and drop NaNs
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train = df_train.dropna()

# Check if data is available
if df_train.empty:
    st.error("❌ No valid data available for the selected ticker.")
else:
    # Forecasting
    model = Prophet()
    model.fit(df_train)
    future = model.make_future_dataframe(periods=period)
    forecast = model.predict(future)

    # Show forecast
    st.subheader("Forecast Data")
    st.write(forecast.tail())

    # Plot forecast
    st.subheader("Forecast Plot")
    fig1 = plot_plotly(model, forecast)
    st.plotly_chart(fig1)

    # Forecast components
    st.subheader("Forecast Components")
    fig2 = model.plot_components(forecast)
    st.write(fig2)
