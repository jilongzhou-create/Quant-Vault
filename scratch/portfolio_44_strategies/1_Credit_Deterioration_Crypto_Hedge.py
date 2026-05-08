import pandas as pd
import numpy as np

def generate_signals(df, params):
    # 安全获取参数，预设合理默认值
    hyc_diff_threshold = params.get('hyc_diff_threshold', 5.0)
    walcl_diff_threshold = params.get('walcl_diff_threshold', -10.0)
    macro_score_threshold = params.get('macro_score_threshold', 2)
    macro_window = params.get('macro_window', 20)
    mvrv_short_window = params.get('mvrv_short_window', 20)
    mvrv_long_window = params.get('mvrv_long_window', 60)
    
    # 1. 垃圾债利差逻辑 (bamlh0a3hyc)
    if 'bamlh0a3hyc' in df.columns and not df['bamlh0a3hyc'].isna().all():
        hyc = df['bamlh0a3hyc'].ffill().fillna(0)
        hyc_180_ma = hyc.rolling(180, min_periods=1).mean()
        hyc_10_diff = hyc.diff(10).fillna(0)
        # 快速飙升且大于过去半年均值
        hyc_cond = (hyc > hyc_180_ma) & (hyc_10_diff > hyc_diff_threshold)
    else:
        hyc_cond = pd.Series(True, index=df.index)  # 缺失降级
        
    # 2. 联储资产规模逻辑 (walcl)
    if 'walcl' in df.columns and not df['walcl'].isna().all():
        walcl = df['walcl'].ffill().fillna(0)
        walcl_30_diff = walcl.diff(30).fillna(0)
        # 停止收缩或转为扩张
        walcl_cond = walcl_30_diff > walcl_diff_threshold
    else:
        walcl_cond = pd.Series(True, index=df.index)  # 缺失降级
        
    # 3. 政策不确定性逻辑 (usepuindxd)
    if 'usepuindxd' in df.columns and not df['usepuindxd'].isna().all():
        epu = df['usepuindxd'].ffill().fillna(0)
        epu_180_ma = epu.rolling(180, min_periods=1).mean()
        # 处于高位(大于半年均线)
        epu_cond = epu > epu_180_ma
    else:
        epu_cond = pd.Series(True, index=df.index)  # 缺失降级
        
    # 宏观综合预警评分机制（防止苛刻单点条件导致死区）
    macro_score = hyc_cond.astype(int) + walcl_cond.astype(int) + epu_cond.astype(int)
    macro_warning = macro_score >= macro_score_threshold
    
    # 预警状态延续，给予资金入场反应时间窗口
    macro_warning_state = macro_warning.rolling(window=macro_window, min_periods=1).max() > 0
    
    # 4. 估值与价格确认逻辑 (mvrv)
    if 'mvrv' in df.columns and not df['mvrv'].isna().all():
        mvrv = df['mvrv'].ffill().fillna(1.0)
        mvrv_top_val = params.get('mvrv_top', 2.5)
    else:
        # 降级：若无mvrv，用价格偏离度替代概念
        mvrv = df['close'] / df['close'].rolling(60, min_periods=1).mean().replace(0, np.nan)
        mvrv = mvrv.fillna(1.0)
        mvrv_top_val = params.get('mvrv_top', 1.4)  # 偏离度过大视为高估
        
    mvrv_short_ma = mvrv.rolling(mvrv_short_window, min_periods=1).mean()
    mvrv_long_ma = mvrv.rolling(mvrv_long_window, min_periods=1).mean()
    
    # 买入条件：处于宏观危机预警期内，且估值指标右侧走强确认资金进场
    buy_condition = macro_warning_state & (mvrv > mvrv_short_ma)
    
    # 平仓条件：
    # 1. 估值跌破长周期均线(长期趋势被破坏，不抱幻想)
    # 2. 宏观危机彻底解除(score=0) 且 短期跌破短均线
    # 3. 极度高估过热且死叉回调(防守止盈)
    sell_condition = (mvrv < mvrv_long_ma) | \
                     ((macro_score == 0) & (mvrv < mvrv_short_ma)) | \
                     ((mvrv > mvrv_top_val) & (mvrv < mvrv_short_ma))
                     
    # 5. 持仓状态生成
    pos = pd.Series(np.nan, index=df.index)
    pos[buy_condition] = 1
    pos[sell_condition] = 0
    
    # 向前填充维持长线持仓，开头填0
    pos = pos.ffill().fillna(0)
    
    return pos
