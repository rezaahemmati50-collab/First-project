import pandas as pd
import yfinance as yf
import numpy as np

# تلاش برای وارد کردن Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# تابع دریافت داده‌ (روزانه)
def fetch_data(symbol, days):
    """
    symbol: e.g. "BTC-USD"
    days: integer
    returns: DataFrame with DatetimeIndex and columns including 'Close'
    """
    try:
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.Timedelta(days=days)
        df = yf.download(symbol, start=start.date(), end=end.date(), progress=False, threads=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()

# محاسبه MA و RSI و ریزه‌کاری‌ها
def compute_indicators(df, ma_periods=(20,50)):
    df = df.copy()
    if "Close" not in df.columns:
        return df
    # MAها
    ma1, ma2 = ma_periods
    df[f"MA{ma1}"] = df["Close"].rolling(window=ma1).mean()
    df[f"MA{ma2}"] = df["Close"].rolling(window=ma2).mean()
    # RSI ساده
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

# پیش‌بینی با Prophet (اگر در محیط نصب باشد)
def predict_prophet(df, days_ahead=3):
    """
    df: DataFrame with DateTimeIndex and 'Close'
    returns: DataFrame with columns ds (date) and yhat (prediction) OR string message on error
    """
    if not PROPHET_AVAILABLE:
        return "Prophet package not installed on server. Install 'prophet' to enable forecasting."
    if df is None or df.empty or "Close" not in df.columns:
        return "Not enough data for forecasting."
    try:
        hist = df[["Close"]].reset_index().rename(columns={"index":"ds","Close":"y"})
        hist["ds"] = pd.to_datetime(hist["ds"])
        hist = hist.dropna()
        if len(hist) < 10:
            return "Not enough historical rows for Prophet forecasting."
        m = Prophet(daily_seasonality=True)
        m.fit(hist)
        future = m.make_future_dataframe(periods=days_ahead)
        forecast = m.predict(future)
        out = forecast[["ds","yhat"]].tail(days_ahead)
        out["ds"] = out["ds"].dt.date
        return out
    except Exception as e:
        return f"Forecast error: {e}"

# تابع نمونه برای گرفتن اخبار (placeholder)
def safe_get_news(name):
    """
    اگر API کلید داشتی می‌تونی این تابع رو گسترش بدی.
    در حال حاضر پیغام مناسب می‌فرستد.
    """
    # اگر می‌خواهی اخبار واقعی بگیری، اینجا requests به NewsAPI یا منابع دیگر بزن
    # برای الان یک پیام راهنما برمی‌گردانیم:
    return []  # لیستی از دیکشنری‌های {title, url, source}
