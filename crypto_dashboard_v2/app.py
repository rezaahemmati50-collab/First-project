import streamlit as st
from data_loader import load_crypto_data
from utils import plot_price_chart
from config import CRYPTO_LIST
import datetime

# Page config
st.set_page_config(page_title="Crypto Dashboard", page_icon="💰", layout="wide")

# Custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.title("💰 Crypto Analysis Dashboard")

# Sidebar
st.sidebar.header("Settings")
selected_crypto = st.sidebar.selectbox("Select a cryptocurrency", CRYPTO_LIST)
start_date = st.sidebar.date_input("Start date", datetime.date(2024, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date.today())

# Load data
df = load_crypto_data(selected_crypto, start_date, end_date)

if df is not None and not df.empty:
    st.subheader(f"Price Chart - {selected_crypto}")
    plot_price_chart(df)
else:
    st.warning("No data available for the selected range.")
