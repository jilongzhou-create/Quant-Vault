import pandas as pd
import numpy as np

def generate_signals(df, params):
    # 1. 参数提取与默认值设置
    sahm_threshold = params.get('sahm_threshold', 1.0)
    t10y3m_threshold = params.get('t10y3m_threshold', 0.0)
    mvrv_cold_threshold = params.get('mvrv_cold_threshold', 1.3)
    macro_bg_window = params.get('macro_bg_window', 90)
    mvrv_ma_fast_win = params.get('mvrv_ma_fast', 10)
    mvrv_ma_slow_win = params.get('mvrv_ma_slow', 30)
    price_ma_win = params.get('price_ma_window', 20)
    mvrv_sell_threshold = params.get('mvrv_sell_threshold', 2.5)
    price_sell_ma_win = params.get('price_sell_ma_window', 100)
    macro_smooth_win = params.get('macro_smooth_window', 30)
    
    # 2. 基础数据预处理与容错
    close = df['close'].ffill()
    
    # 宏观数据提取与平滑 (使用 rolling 极力过滤噪音，若因子缺失则启用容错降级)
    if 'sahmrealtime' in df.columns:
        sahm_raw = df['sahmrealtime'].ffill().fillna(0)
        # 量纲自适应：若数据为百分比放大100倍形式，则自适应调整阈值
        sahm_thresh_adj = sahm_threshold * 100 if sahm_raw.max() > 10 else sahm_threshold
        sahm = sahm_raw.rolling(window=macro_smooth_win, min_periods=1).mean()
        has_sahm = sahm_raw.notna().cumsum() > 0
    else:
        sahm = pd.Series(0, index=df.index)
        sahm_thresh_adj = sahm_threshold
        has_sahm = pd.Series(False, index=df.index)
        
    if 't10y3m' in df.columns:
        t10_raw = df['t10y3m'].ffill().fillna(-1.0)
        t10_thresh_adj = t10y3m_threshold * 100 if t10_raw.max() > 10 else t10y3m_threshold
        t10y3m = t10_raw.rolling(window=macro_smooth_win, min_periods=1).mean()
        has_t10 = t10_raw.notna().cumsum() > 0
    else:
        t10y3m = pd.Series(0, index=df.index)
        t10_thresh_adj = t10y3m_threshold
        has_t10 = pd.Series(False, index=df.index)
        
    if 'mvrv' in df.columns:
        mvrv_raw = df['mvrv'].ffill().fillna(1.5)
        mvrv_cold_adj = mvrv_cold_threshold * 100 if mvrv_raw.max() > 10 else mvrv_cold_threshold
        mvrv_sell_adj = mvrv_sell_threshold * 100 if mvrv_raw.max() > 10 else mvrv_sell_threshold
        mvrv = mvrv_raw
        has_mvrv = mvrv_raw.notna().cumsum() > 0
    else:
        mvrv = pd.Series(1.5, index=df.index)
        mvrv_cold_adj = mvrv_cold_threshold
        mvrv_sell_adj = mvrv_sell_threshold
        has_mvrv = pd.Series(False, index=df.index)

    # 3. 宏观左侧底与估值左侧底条件计算
    # 若缺失该宏观变量，容错机制允许忽略该条件，防止死区
    cond_sahm = (sahm > sahm_thresh_adj) | (~has_sahm)
    cond_t10 = (t10y3m > t10_thresh_adj) | (~has_t10)
    cond_mvrv_cold = (mvrv < mvrv_cold_adj) | (~has_mvrv)
    
    # 构建宏观背景：最近 macro_bg_window 天内曾经满足过衰退与估值极寒
    macro_condition = cond_sahm & cond_t10 & cond_mvrv_cold
    macro_background = macro_condition.rolling(window=macro_bg_window, min_periods=1).max() > 0
    
    # 4. 技术面与链上的右侧共振信号计算
    mvrv_fast = mvrv.rolling(window=mvrv_ma_fast_win, min_periods=1).mean()
    mvrv_slow = mvrv.rolling(window=mvrv_ma_slow_win, min_periods=1).mean()
    price_ma = close.rolling(window=price_ma_win, min_periods=1).mean()
    price_sell_ma = close.rolling(window=price_sell_ma_win, min_periods=1).mean()
    
    # 采用状态组合，提取右侧突破瞬间
    mvrv_is_up = mvrv_fast > mvrv_slow
    price_is_up = close > price_ma
    
    right_side_state = mvrv_is_up & price_is_up
    # 前一天未进入多头状态，而今日完全满足，即为右侧触发日
    right_side_trigger = right_side_state & (~right_side_state.shift(1).fillna(False))
    
    buy_condition = macro_background & right_side_trigger
    
    # 5. 卖出与止损保护逻辑 (防震荡洗盘)
    # a. 链上估值达到绝对高位泡沫区间
    sell_mvrv_high = mvrv > mvrv_sell_adj
    
    # b. 右侧趋势彻底反转（估值高位死叉 且 标的跌破长期生命线）
    mvrv_is_down = mvrv_fast < mvrv_slow
    price_is_down = close < price_sell_ma
    sell_trend_reversal = mvrv_is_down & price_is_down
    
    sell_condition = sell_mvrv_high | sell_trend_reversal
    
    # 6. 生成目标持仓状态
    signals = pd.Series(np.nan, index=df.index)
    signals[buy_condition] = 1
    signals[sell_condition] = 0
    
    # 向前填充维持长线持仓，未触发交易的初期填补0
    signals = signals.ffill().fillna(0)
    
    return signals
