@st.cache_data(ttl=180)
def fetch_yf(symbol: str, period="3mo", interval="1d") -> pd.DataFrame:
    import concurrent.futures
    
    try:
        # enforce timeout (3s)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                yf.download, symbol, period=period, interval=interval, progress=False
            )
            df = future.result(timeout=3)  # max 3 seconds wait
        if df is None or df.empty:
            st.warning(f"No Yahoo data for {symbol}, using fake data.")
            return fake_data(symbol)
        return normalize_ohlc_index(df)
    except Exception:
        st.warning(f"Yahoo Finance error for {symbol}, using fake data.")
        return fake_data(symbol)
