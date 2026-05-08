def generate_signals(df, params):
    import pandas as pd
    import numpy as np

    # 1. 提取参数，设置安全的默认值
    nfci_window = params.get('nfci_window', 28)
    nasdaq_window = params.get('nasdaq_window', 20)
    nfci_diff_th = params.get('nfci_diff_th', 0.0)
    nasdaq_mom_th = params.get('nasdaq_mom_th', 0.0)
    sma_fast_period = params.get('sma_fast', 20)
    sma_slow_period = params.get('sma_slow', 60)
    dev_threshold = params.get('dev_threshold', 0.08)
    short_dev_th = params.get('short_dev_th', 0.12)
    
    # 2. 因子防缺失降级与平滑处理
    # 很多宏观指标在早期为NaN，使用0做中性替代以防止逻辑中断归零
    if 'nfci' in df.columns:
        nfci = df['nfci'].ffill().fillna(0)
    else:
        nfci = pd.Series(0, index=df.index)
        
    if 'nasdaqcom' in df.columns:
        nasdaq = df['nasdaqcom'].ffill().fillna(0)
    else:
        nasdaq = pd.Series(0, index=df.index)
        
    # NFCI宏观趋势变动：计算4周均线，并求4周之差（正值表示金融条件边际收紧）
    nfci_smooth = nfci.rolling(window=nfci_window, min_periods=1).mean()
    nfci_diff = nfci_smooth.diff(nfci_window).fillna(0)
    
    # Nasdaq动量趋势：5日平滑去噪后计算20日边际变化（负值表示美股走弱）
    nasdaq_smooth = nasdaq.rolling(window=5, min_periods=1).mean()
    nasdaq_mom = nasdaq_smooth.diff(nasdaq_window).fillna(0)
    
    # 判定当前环境是否为宏观逆风 (Headwind)
    is_headwind = (nfci_diff > nfci_diff_th) & (nasdaq_mom < nasdaq_mom_th)
    
    # 3. 基础量价因子与均线判定
    close = df['close']
    sma_fast = close.rolling(window=sma_fast_period, min_periods=1).mean()
    sma_slow = close.rolling(window=sma_slow_period, min_periods=1).mean()
    # 当前价格距离快线的偏离度
    price_dev = (close - sma_fast) / (sma_fast + 1e-8)
    
    # 4. 构建宏观约束下的买卖核心逻辑
    # 【做多逻辑】：非逆风且处于长线多头趋势中
    buy_cond = (~is_headwind) & (close > sma_slow) & (sma_fast >= sma_slow)
    
    # 【平多逻辑】：逆风且价格过度向上偏离，或跌破长期趋势支撑
    close_long_cond = (close < sma_slow) | (is_headwind & (price_dev > dev_threshold))
    
    # 【做空逻辑】：宏观极度逆风且价格严重向上偏离（超买反转）
    short_cond = is_headwind & (price_dev > short_dev_th)
    
    # 【平空逻辑】：价格向均线修复或宏观逆风压制解除
    close_short_cond = (price_dev < 0) | (~is_headwind)
    
    # 5. 信号整合与状态机持仓保持
    signals = pd.Series(np.nan, index=df.index)
    
    # 优先标注明确的开仓动作
    signals[buy_cond] = 1
    signals[short_cond] = -1
    
    # 标注平仓动作 (通过排他组合避免将刚建立的反向仓位误平)
    signals[close_long_cond & ~short_cond] = 0
    signals[close_short_cond & ~buy_cond] = 0
    
    # ffill() 向前填充以维持中长期持仓，初始时以0空仓起始
    signals = signals.ffill().fillna(0)
    
    return signals