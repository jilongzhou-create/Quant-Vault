import numpy as np
import pandas as pd

def generate_signals(df, params):
    """
    极度鲁棒的宏观长线策略：链上MVRV与情绪背离
    """
    # 1. 提取参数，设定宽松但有效的宏观阈值
    mvrv_buy = params.get('mvrv_buy', 1.2)
    fg_buy = params.get('fg_buy', 35.0)
    mvrv_sell = params.get('mvrv_sell', 2.5)
    fr_sell = params.get('fr_sell', 0.00015)
    smooth_window = params.get('smooth_window', 30)
    enable_short = params.get('enable_short', True)  # 顶部是否反手做空

    # 2. 初始化目标持仓信号
    signals = pd.Series(np.nan, index=df.index)

    # 3. 因子防缺失降级处理 (Fallback)，防止NaN导致策略整体归零
    if 'mvrv' in df.columns:
        mvrv = df['mvrv'].ffill().fillna(1.5)  # 默认1.5为中性估值
    else:
        mvrv = pd.Series(1.5, index=df.index)
        
    if 'fear_greed' in df.columns:
        fg = df['fear_greed'].ffill().fillna(50.0)  # 默认50为中性情绪
    else:
        fg = pd.Series(50.0, index=df.index)
        
    if 'funding_rate' in df.columns:
        fr = df['funding_rate'].ffill().fillna(0.0)  # 默认0为现货/永续无杠杆偏差
    else:
        fr = pd.Series(0.0, index=df.index)

    # 4. 宏观噪音平滑（极度重要：过滤短线震荡，提纯主趋势）
    mvrv_ma = mvrv.rolling(window=smooth_window, min_periods=1).mean()
    fg_ma = fg.rolling(window=smooth_window, min_periods=1).mean()
    fr_ma = fr.rolling(window=smooth_window, min_periods=1).mean()

    # 5. 核心交易逻辑（极简、鲁棒）
    # 建仓条件：链上深度低估 + 宏观情绪极度恐慌
    buy_cond = (mvrv_ma < mvrv_buy) & (fg_ma < fg_buy)
    
    # 平仓/做空条件：链上严重高估 + 衍生品多头杠杆过热
    sell_cond = (mvrv_ma > mvrv_sell) & (fr_ma > fr_sell)

    # 6. 持仓状态赋值
    signals[buy_cond] = 1
    signals[sell_cond] = -1 if enable_short else 0

    # 7. 坚定持仓：使用向前填充，只要不触发反转信号，死死拿住宏观大波段
    signals = signals.ffill().fillna(0)

    return signals
