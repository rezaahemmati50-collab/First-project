# app.py (robust version)
import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import date

st.set_page_config(page_title="Crypto Forecast (robust)", layout="wide")
st.title("📈 Crypto Forecast (robust)")

# user input
ticker_input = st.text_input("Enter ticker (e.g. BTC-USD). For multiple, separate with comma:", "BTC-USD")
n_days = st.slider("Forecast days:", 1, 365, 30)

# normalize ticker: if user provided a comma separated list, take first as primary ticker
tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
primary_ticker = tickers[0] if len(tickers) > 0 else ticker_input.strip()

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

@st.cache_data(ttl=300)
def download_data(tickers):
    """
    Use yfinance.download. If user passed multiple tickers, yfinance returns MultiIndex columns.
    We just return whatever yfinance gives and handle it later.
    """
    try:
        df = yf.download(tickers, start=START, end=TODAY, progress=False)
        if isinstance(df, pd.DataFrame):
            df = df.copy()
        return df
    except Exception as e:
        st.error(f"Failed to download data: {e}")
        return pd.DataFrame()

df = download_data(ticker_input)

# quick debug info (optional)
st.write("### Debug: columns received")
st.write(list(df.columns)[:20])
# show head
st.write("### Data preview")
st.write(df.head())

# ---------- robust extraction of Close series ----------
def extract_close_series(df, primary_ticker=None):
    """
    Return a pandas Series representing the Close prices for the selected ticker.
    Handles:
      - normal DataFrame with column 'Close'
      - DataFrame with MultiIndex columns (e.g. (ticker, 'Close'))
      - DataFrame where df['Close'] returns a DataFrame (multiple tickers) -> pick matching column
    Returns: (series, message) where series is pd.Series or None and message is info/error string.
    """
    if df is None or df.empty:
        return None, "No data returned from yfinance."

    # If columns are MultiIndex (e.g. multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        # try to get the 'Close' level (usually level=1)
        try:
            # This returns a DataFrame with columns = top-level tickers
            close_df = df.xs('Close', axis=1, level=1)
            # if only one ticker -> return its series
            if isinstance(close_df, pd.Series):
                # xs can return Series if only one column exists
                return close_df, "Close column extracted from MultiIndex (single column)."
            if close_df.shape[1] == 1:
                return close_df.iloc[:, 0], "Close column extracted (single ticker)."
            # multiple tickers' close columns available
            if primary_ticker and primary_ticker in close_df.columns:
                return close_df[primary_ticker], f"Close for {primary_ticker} selected from multi-close DataFrame."
            # fallback: pick the first available ticker column
            return close_df.iloc[:, 0], f"Multiple Close columns found; using first column ({close_df.columns[0]})."
        except KeyError:
            return None, "MultiIndex present but no 'Close' level found."
        except Exception as e:
            return None, f"Error extracting Close from MultiIndex: {e}"

    # If normal columns
    if 'Close' in df.columns:
        close_col = df['Close']
        # If df['Close'] returns a DataFrame (rare), handle it
        if isinstance(close_col, pd.DataFrame):
            # multiple columns for Close (maybe multiple tickers); try primary_ticker
            if primary_ticker and primary_ticker in close_col.columns:
                return close_col[primary_ticker], f"Close for {primary_ticker} selected from df['Close'] DataFrame."
            # fallback to first column
            return close_col.iloc[:, 0], "df['Close'] is a DataFrame; using its first column."
        # if it's a Series — perfect
        if isinstance(close_col, pd.Series):
            return close_col, "Close column is a Series."
        # if it's list-like
        if isinstance(close_col, (list, tuple)):
            return pd.Series(close_col), "Close was list/tuple converted to Series."
        # unknown type
        return None, f"df['Close'] exists but has unexpected type: {type(close_col)}"

    # some dataframes might call it 'Adj Close'
    if 'Adj Close' in df.columns:
        adj = df['Adj Close']
        if isinstance(adj, pd.Series):
            return adj, "Using 'Adj Close' as fallback."
        if isinstance(adj, pd.DataFrame):
            return adj.iloc[:,0], "Using first column of 'Adj Close' DataFrame as fallback."

    return None, "No Close or Adj Close column found."

close_series, msg = extract_close_series(df, primary_ticker=primary_ticker)
st.write("Extraction message:", msg)

if close_series is None:
    st.error("❌ Could not extract a Close price series. See debug info above. Possible reasons: wrong ticker, multiple tickers without specifying one, or yfinance returned no data.")
    st.stop()

# ensure close_series is pandas Series
if not isinstance(close_series, pd.Series):
    close_series = pd.Series(close_series)

# convert to numeric safely
try:
    close_series = pd.to_numeric(close_series, errors='coerce')
except Exception as e:
    st.error(f"Error converting Close to numeric: {e}")
    st.stop()

if close_series.isna().all():
    st.error("❌ All Close values are non-numeric after conversion.")
    st.stop()

# Build df_train for Prophet
# If original df has Date column use it; else use index
if 'Date' in df.columns:
    ds = pd.to_datetime(df['Date'], errors='coerce')
else:
    ds = pd.to_datetime(df.index, errors='coerce')

# align lengths in case we took a column from a sub-DataFrame
ds = ds.reset_index(drop=True)
close_series = close_series.reset_index(drop=True)

df_train = pd.DataFrame({'ds': ds, 'y': close_series}).dropna(subset=['ds','y'])
if df_train.empty:
    st.error("❌ After alignment & cleaning, no valid rows remain for forecasting.")
    st.stop()

# Plot raw close
st.subheader("Raw Close series (used for forecasting)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], name="Close"))
fig.update_layout(xaxis_rangeslider_visible=True, height=400)
st.plotly_chart(fig, use_container_width=True)

# Prophet forecasting
st.subheader("Forecast")
model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=n_days)
forecast = model.predict(future)

fig_forecast = plot_plotly(model, forecast)
st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("Forecast table (tail)")
st.write(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail())

st.success("Done ✅")
