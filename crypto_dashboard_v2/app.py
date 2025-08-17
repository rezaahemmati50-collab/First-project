import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.title("Crypto Dashboard - Offline Mode")

# ساخت دیتای تستی برای ۳۰ روز گذشته
dates = pd.date_range(end=datetime.today(), periods=30)
prices = np.linspace(25000, 30000, 30) + np.random.normal(0, 500, 30)

df = pd.DataFrame({"Date": dates, "Price": prices})

st.line_chart(df.set_index("Date")["Price"])

st.success("Fake data loaded successfully (no API calls).")
