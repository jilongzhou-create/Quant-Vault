import pandas as pd
import numpy as np

def generate_signals(df, params):
    # 安全提取策略参数
    dfii10_ma_win = params.get('dfii10_ma_win', 50)
    dfii10_smooth_win = params.get('dfii10_smooth_win', 10)
    t10y2y_roc_win = params.get('t10y2y_roc_win', 30)
    t10y2y_spread_thresh = params.get('t10y2y_spread_thresh', 5.0)
    
    # 1. 因子防缺失与填充降级处理
    dfii10 = df['dfii10'].ffill().fillna(0) if 'dfii10' in df.columns else pd.Series(0, index=df.index)
    t10y2y = df['t10y2y'].ffill().fillna(0) if 't10y2y' in df.columns else pd.Series(0, index=df.index)
    
    # 判断因子是否全面缺失（例如无有效波动），若缺失则启用 Fallback
    use_fallback = (dfii10.std() < 1e-5) or (t10y2y.std() < 1e-5)
    
    # 2. 宏观指标计算与极限平滑过滤
    dfii10_smooth = dfii10.rolling(window=dfii10_smooth_win, min_periods=1).mean()
    dfii10_ma = dfii10.rolling(window=dfii10_ma_win, min_periods=1).mean()
    
    t10y2y_smooth = t10y2y.rolling(window=10, min_periods=1).mean()
    # 计算期限利差的走阔/收窄动量
    t10y2y_roc = t10y2y_smooth - t10y2y_smooth.shift(t10y2y_roc_win).fillna(0)
    
    # 3. 构建宏观基本面触发逻辑
    # Risk-off 做空条件：实际利率向上突破均线 且 期限利差明显走阔
    cond_risk_off = (dfii10_smooth > dfii10_ma) & (t10y2y_roc > t10y2y_spread_thresh)
    
    # Risk-on 做多条件：实际利率回落至均线下方 且 期限利差明显收窄
    cond_risk_on = (dfii10_smooth < dfii10_ma) & (t10y2y_roc < -t10y2y_spread_thresh)
    
    # 中性平仓条件：两者方向发生背离，宏观进入混沌期
    cond_neutral = (
        ((dfii10_smooth < dfii10_ma) & (t10y2y_roc > 0)) | 
        ((dfii10_smooth > dfii10_ma) & (t10y2y_roc < 0))
    )
    
    # 4. Fallback 备用策略：在无宏观数据时依靠低频价量均线
    if use_fallback and 'close' in df.columns:
        close = df['close']
        close_fast = close.rolling(window=50, min_periods=1).mean()
        close_slow = close.rolling(window=120, min_periods=1).mean()
        buy_condition = close_fast > close_slow
        sell_condition = close_fast < close_slow
        neutral_condition = pd.Series(False, index=df.index)
    else:
        buy_condition = cond_risk_on
        sell_condition = cond_risk_off
        neutral_condition = cond_neutral
        
    # 5. 信号整合与持仓保持
    signals = pd.Series(np.nan, index=df.index)
    signals[buy_condition] = 1
    signals[sell_condition] = -1
    signals[neutral_condition] = 0
    
    # 前向填充以锁定宏观长线主升/主跌浪，避免高频颠簸
    signals = signals.ffill().fillna(0)
    
    return signals