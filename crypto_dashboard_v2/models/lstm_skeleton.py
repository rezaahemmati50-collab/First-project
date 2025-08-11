# models/lstm_skeleton.py
"""
LSTM skeleton (template). NOT enabled by default.
To use: install tensorflow and adapt train/load code.
"""
import numpy as np
import pandas as pd

def prepare_series_for_lstm(series, lookback=30):
    s = series.dropna().astype(float)
    X, y = [], []
    for i in range(lookback, len(s)):
        X.append(s.iloc[i-lookback:i].values.reshape(-1,1))
        y.append(s.iloc[i])
    return np.array(X), np.array(y)
