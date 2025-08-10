import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import os

# عنوان برنامه
st.title("📈 پیش‌بینی قیمت ارز دیجیتال با Prophet")

# مسیر فایل نمونه دیتا
sample_path = os.path.join("data", "sample.csv")

# بارگذاری دیتا
uploaded_file = st.file_uploader("یک فایل CSV آپلود کنید یا از نمونه استفاده کنید", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv(sample_path)
    st.info("⚠️ از داده نمونه استفاده شد.")

# بررسی ستون‌ها
st.subheader("دیتای ورودی")
st.write(df.head())

if 'ds' not in df.columns or 'y' not in df.columns:
    st.error("فایل باید شامل دو ستون `ds` (تاریخ) و `y` (قیمت) باشد.")
    st.stop()

# تبدیل تاریخ
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

# اطمینان از عددی بودن y
if not pd.api.types.is_numeric_dtype(df['y']):
    try:
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
    except Exception as e:
        st.error(f"❌ خطا در تبدیل ستون y به عدد: {e}")
        st.stop()

# حذف ردیف‌های خالی
df = df.dropna(subset=['ds', 'y'])

# آموزش مدل Prophet
model = Prophet()
model.fit(df)

# پیش‌بینی ۳۰ روز آینده
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# نمایش نتیجه
fig1 = px.line(forecast, x='ds', y='yhat', title="📊 پیش‌بینی قیمت")
st.plotly_chart(fig1)

st.subheader("نمونه داده پیش‌بینی")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
