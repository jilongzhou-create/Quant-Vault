import pandas as pd
import numpy as np

def generate_signals(df, params):
    # 安全提取参数，设置适应长线交易的宏观默认值
    vol_window = params.get('vol_window', 20)
    trend_window = params.get('trend_window', 60)
    long_trend_window = params.get('long_trend_window', 120)
    range_window = params.get('range_window', 30)
    range_mult = params.get('range_mult', 1.5)
    macro_window = params.get('macro_window', 30)
    nfci_buy_z = params.get('nfci_buy_z', 1.0)
    mvrv_buy_z = params.get('mvrv_buy_z', 1.5)
    nfci_sell_z = params.get('nfci_sell_z', 2.0)
    mvrv_sell_z = params.get('mvrv_sell_z', 2.5)
    trend_buy_margin = params.get('trend_buy_margin', 1.01)
    trend_sell_margin = params.get('trend_sell_margin', 0.95)
    vol_ratio_max = params.get('vol_ratio_max', 1.5)
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # --- 1. 宏观/另类基本面因子的处理与降级 (防缺失 & 防未来函数) ---
    # nfci: 金融环境指数。正值表示紧缩，负值宽松。通过滚动Z-Score避免量纲灾难。
    nfci = df['nfci'].ffill().fillna(0) if 'nfci' in df.columns else pd.Series(0, index=df.index)
    nfci_mean = nfci.rolling(180, min_periods=1).mean()
    nfci_std = nfci.rolling(180, min_periods=1).std().replace(0, 1e-6).fillna(1e-6)
    nfci_z = (nfci - nfci_mean) / nfci_std
    nfci_z_smooth = nfci_z.rolling(macro_window, min_periods=1).mean()
    
    # mvrv: 估值压力指标。通过滚动Z-Score衡量当前泡沫严重程度
    mvrv = df['mvrv'].ffill().fillna(150.0) if 'mvrv' in df.columns else pd.Series(150.0, index=df.index)
    mvrv_mean = mvrv.rolling(180, min_periods=1).mean()
    mvrv_std = mvrv.rolling(180, min_periods=1).std().replace(0, 1e-6).fillna(1e-6)
    mvrv_z = (mvrv - mvrv_mean) / mvrv_std
    mvrv_z_smooth = mvrv_z.rolling(macro_window, min_periods=1).mean()
    
    # 宏观环境条件判断
    buy_macro = (nfci_z_smooth < nfci_buy_z) & (mvrv_z_smooth < mvrv_buy_z)
    sell_macro = (nfci_z_smooth > nfci_sell_z) | (mvrv_z_smooth > mvrv_sell_z)
    
    # --- 2. 价格趋势因子 (Trend) ---
    ma_trend = close.rolling(trend_window, min_periods=1).mean()
    ma_long = close.rolling(long_trend_window, min_periods=1).mean()
    
    # 中期趋势确认上行
    buy_trend = close > (ma_trend * trend_buy_margin)
    # 极力过滤震荡：仅当跌破长线支撑一定深度才认定趋势破位卖出
    sell_trend = close < (ma_long * trend_sell_margin)
    
    # --- 3. 波动率因子 (Volatility) ---
    # 计算对数收益率历史波动率，规避短期插针带来的极值污染
    log_ret = np.log(close / close.shift(1).replace(0, np.nan)).fillna(0)
    log_ret = log_ret.where(log_ret.abs() <= 0.5, 0)
    volatility = log_ret.rolling(vol_window, min_periods=1).std().fillna(0)
    vol_median = volatility.rolling(180, min_periods=1).median().fillna(0)
    
    # 买入前要求标的历史波动平稳，未陷入杂乱无章的高波洗盘
    buy_vol = volatility < (vol_median * vol_ratio_max)
    
    # --- 4. 极差因子 (Range) ---
    daily_range = (high - low) / (close + 1e-8)
    range_ma = daily_range.rolling(range_window, min_periods=1).mean().fillna(0)
    # 捕捉当日振幅显著大于长周期均值的动量爆发点
    buy_range = daily_range > (range_ma * range_mult)
    
    # --- 5. 信号生成与持仓保持 (State Holding) ---
    buy_condition = buy_macro & buy_trend & buy_vol & buy_range
    sell_condition = sell_trend | sell_macro
    
    signals = pd.Series(np.nan, index=df.index)
    signals[buy_condition] = 1
    signals[sell_condition] = 0
    
    # 坚守宏观波段，只要卖出条件未触发，通过前向填充坚定持仓
    signals = signals.ffill().fillna(0)
    
    return signals
