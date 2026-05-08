import pandas as pd
import numpy as np

def generate_signals(df, params):
    zscore_window = params.get('zscore_window', 60)
    smooth_window = params.get('smooth_window', 5)
    fedfunds_window = params.get('fedfunds_window', 30)
    fedfunds_stable_threshold = params.get('fedfunds_stable_threshold', 20.0)
    buy_zscore_threshold = params.get('buy_zscore_threshold', 1.8)
    sell_zscore_threshold = params.get('sell_zscore_threshold', 0.0)
    
    if 'dfii10' in df.columns and not df['dfii10'].isna().all():
        real_yield = df['dfii10'].ffill().fillna(0)
    elif 'dgs10' in df.columns and not df['dgs10'].isna().all():
        real_yield = df['dgs10'].ffill().fillna(0)
    else:
        return pd.Series(0, index=df.index)
        
    if 'fedfunds' in df.columns and not df['fedfunds'].isna().all():
        fedfunds = df['fedfunds'].ffill().fillna(0)
    else:
        fedfunds = pd.Series(0, index=df.index)
        
    ry_smoothed = real_yield.rolling(window=smooth_window, min_periods=1).mean()
    ry_mean = ry_smoothed.rolling(window=zscore_window, min_periods=1).mean()
    ry_std = ry_smoothed.rolling(window=zscore_window, min_periods=1).std()
    ry_zscore = (ry_smoothed - ry_mean) / (ry_std + 1e-6)
    
    ff_std = fedfunds.rolling(window=fedfunds_window, min_periods=1).std().fillna(0)
    fedfunds_stable = ff_std < fedfunds_stable_threshold
    
    buy_condition = (ry_zscore > buy_zscore_threshold) & fedfunds_stable
    sell_condition = ry_zscore < sell_zscore_threshold
    
    signals = pd.Series(np.nan, index=df.index)
    signals[buy_condition] = 1
    signals[sell_condition] = 0
    signals = signals.ffill().fillna(0)
    
    return signals
