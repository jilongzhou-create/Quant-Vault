import numpy as np
import pandas as pd

class EpuPanicMeanReversionFactor:
    """经济政策不确定性恐慌均值回归因子 (panic_mean_reversion/unstructured)

    逻辑: 采用非结构化新闻提取的经济政策不确定性指数(EPU)作为恐慌代理。对于美股(SPY)而言，"利空落地即利好"，当EPU飙升至历史一年期极高点(极端恐慌)并在次日开始回落(恐慌衰竭)时，产生精准的抄底看多脉冲；当EPU处于轻度高位(Z-Score 0.8~2.0)且呈现二阶加速上升时，市场处于钝刀割肉的发酵期，产生看空脉冲。
    数据: usepuindxd (Daily US Economic Policy Uncertainty Index, 基于NLP新闻提取的宏观不确定性指数)
    输出: +1.0 (恐慌极值见顶回落，强烈看多), -1.0 (轻度恐慌加速发酵，顺势看空), 0.0 (常态休眠)
    触发条件: 平滑EPU Z-score>2.0且出现严格峰值拐点触发多头(持仓3天)，Z-score[0.8, 2.0]且二阶导>0触发空头(持仓2天)，预期 Trigger Rate 8-12%
    """

    def __init__(self):
        self.name = 'epu_panic_mean_reversion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少所需数据，直接返回常态休眠信号 0.0
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        # 1. 提取并平滑EPU指数 
        # (EPU日频新闻数据受单日热点影响噪音极大，使用5日EMA/一个交易周提取基础趋势)
        epu = data['usepuindxd'].ffill()
        epu_smooth = epu.ewm(span=5, adjust=False).mean()
        
        # 2. 计算一年期(252日)滚动Z-Score，衡量当前不确定性在历史周期中的相对极值
        epu_mean = epu_smooth.rolling(window=252, min_periods=63).mean()
        epu_std = epu_smooth.rolling(window=252, min_periods=63).std()
        zscore = (epu_smooth - epu_mean) / (epu_std + 1e-6)
        
        # 3. 计算一阶导与二阶导 (边际变化与动量加速)
        epu_diff = epu_smooth.diff(1)
        epu_diff_prev = epu_diff.shift(1)
        
        # 4. 买入脉冲 (二阶导数铁律: 极值 + 衰竭)
        # 极端恐慌(Z-Score > 2.0) 且 昨日还在上升、今日开始下降 (严格捕捉恐慌见顶的瞬间)
        buy_trigger = (zscore > 2.0) & (epu_diff < 0) & (epu_diff_prev > 0)
        
        # 5. 卖出脉冲 (轻度恐慌恶化)
        # 情绪处于轻度恐慌(0.8 < zscore <= 2.0) 且 正在加速上升(一阶导>0 且 今日增幅大于昨日增幅)
        sell_trigger = (zscore > 0.8) & (zscore <= 2.0) & (epu_diff > 0) & (epu_diff > epu_diff_prev)
        
        # 6. 脉冲展期 (将瞬间的Trigger展期极短的几天，以捕捉短期趋势并控制Trigger Rate在 5%-15%)
        # 抄底信号属于左侧极值，给足3天窗口期；空头信号属于动量发酵，给2天窗口期
        buy_signal = buy_trigger.rolling(window=3, min_periods=1).max() > 0
        sell_signal = sell_trigger.rolling(window=2, min_periods=1).max() > 0
        
        # 7. 信号合成
        signal = pd.Series(0.0, index=data.index)
        signal[buy_signal] = 1.0
        signal[sell_signal] = -1.0
        
        # 冲突处理: 极端罕见情况下若同日发生冲突，强制休眠防接飞刀
        conflict = buy_signal & sell_signal
        signal[conflict] = 0.0
        
        # 确保输出范围严格在 [-1.0, 1.0] 且有名字
        return signal.fillna(0.0).astype(float).rename(self.name)