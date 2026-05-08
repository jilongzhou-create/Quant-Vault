def generate_signals(df, params):
    import pandas as pd
    import numpy as np

    # 1. 安全提取策略参数
    z_window = params.get('z_window', 20)
    macro_window = params.get('macro_window', 30)
    
    fg_short = params.get('fg_short', 75)
    z_short = params.get('z_short', 1.8) # 默认略微放宽阈值避免零交易死区
    fg_long = params.get('fg_long', 25)
    z_long = params.get('z_long', -1.8)
    
    mvrv_buy = params.get('mvrv_buy', 1.8)
    mvrv_sell = params.get('mvrv_sell', 2.8)

    # 2. 核心因子防错提取与降级回退 (因子缺失降级法则)
    mvrv = df.get('mvrv', pd.Series(1.5, index=df.index)).ffill().fillna(1.5)
    fear_greed = df.get('fear_greed', pd.Series(50, index=df.index)).ffill().fillna(50)
    funding_rate = df.get('funding_rate', pd.Series(0.0, index=df.index)).ffill().fillna(0.0)

    # 3. 宏观长线底仓计算 (Macro Fund Manager 工作模式)
    # 极力过滤震荡，大量使用平滑获取数月跨年级别的周期状态
    mvrv_macro = mvrv.rolling(window=macro_window, min_periods=1).mean()
    macro_buy_cond = mvrv_macro < mvrv_buy
    macro_sell_cond = mvrv_macro > mvrv_sell
    
    macro_signal = pd.Series(np.nan, index=df.index)
    macro_signal.loc[macro_buy_cond] = 1
    macro_signal.loc[macro_sell_cond] = -1
    macro_signal = macro_signal.ffill().fillna(0)

    # 4. 周末散户情绪衰退反转 (Weekend Retail Perp Exuberance Fade)
    # 转化绝对值为相对指标，防范量纲灾难
    fr_mean = funding_rate.rolling(window=z_window, min_periods=1).mean()
    fr_std = funding_rate.rolling(window=z_window, min_periods=1).std().replace(0, 1e-5)
    fr_zscore = (funding_rate - fr_mean) / fr_std
    
    # 鲁棒判断时间节点 (防范 df.index 并非 DatetimeIndex 的极端异常)
    is_friday = pd.Series(False, index=df.index)
    is_monday = pd.Series(False, index=df.index)
    if hasattr(df.index, 'dayofweek'):
        is_friday = df.index.dayofweek == 4
        is_monday = df.index.dayofweek == 0
        
    weekend_short_cond = is_friday & (fr_zscore > z_short) & (fear_greed > fg_short)
    weekend_long_cond = is_friday & (fr_zscore < z_long) & (fear_greed < fg_long)
    
    # 周末 Alpha 覆盖层
    weekend_override = pd.Series(np.nan, index=df.index)
    weekend_override.loc[weekend_short_cond] = -1
    weekend_override.loc[weekend_long_cond] = 1
    weekend_override.loc[is_monday] = 0 # 周一清空短线干预，准备回归底仓
    weekend_override = weekend_override.ffill().fillna(0)

    # 5. 信号无缝融合
    # 当处于周末极端反转状态时采用 override 信号进行套利，其余时间坚定持有宏观趋势底仓
    final_signal = np.where(weekend_override != 0, weekend_override, macro_signal)
    
    return pd.Series(final_signal, index=df.index).fillna(0).astype(int)