import numpy as np
import pandas as pd

class RealYieldBreakevenPivotFactor:
    """实际利率与通胀预期剪刀差 (policy_pivot/nonlinear)

    逻辑: 通过10年期实际利率(DFII10)和盈亏平衡通胀预期(T10YIE)的非线性背离捕捉美联储政策转向。
          对于长牛的标普500而言，当实际利率大幅下行且通胀预期回升时，代表典型的'金融条件宽松/鸽派转向'，强烈看多；
          当实际利率大幅上行且通胀预期回落时，代表'流动性收紧/鹰派冲击'，看空。
          同时，因子内置了极端流动性枯竭(如2020年3月Dash for Cash导致的实际利率变态飙升)的均值回归抄底保护。
    数据: dfii10, t10yie
    输出: +1.0 (鸽派转向/恐慌衰竭看多), -1.0 (鹰派紧缩看空)
    触发条件: 实际利率与通胀预期5日边缘动量呈现显著反向，预期Trigger Rate ~ 8-12%
    """

    def __init__(self):
        self.name = 'real_yield_breakeven_pivot'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需字段是否存在
        if 'dfii10' not in data.columns or 't10yie' not in data.columns:
            return signal
            
        # 前向填充缺失值
        real_yield = data['dfii10'].ffill()
        breakeven = data['t10yie'].ffill()
        
        # 计算5日变动量 (捕捉边际动量脉冲)
        ry_diff5 = real_yield.diff(5)
        be_diff5 = breakeven.diff(5)
        
        # 1. 鸽派转向脉冲 (Financial Conditions Easing)
        # 逻辑：实际利率大幅下降(>12bps)，且通胀预期反弹(>2bps)，此时分母端压力骤减，美股Risk-On
        dovish_easing = (ry_diff5 < -0.12) & (be_diff5 > 0.02)
        
        # 2. 鹰派冲击脉冲 (Financial Conditions Tightening)
        # 逻辑：实际利率大幅飙升(>12bps)，且通胀预期回落(<-2bps)，杀估值且伴随轻微恐慌，美股看空
        hawkish_tightening = (ry_diff5 > 0.12) & (be_diff5 < -0.02)
        
        # 3. 极端流动性恐慌衰竭 (Dash for Cash Exhaustion)
        # 防接飞刀二阶导逻辑：实际利率极端飙升(5天内>35bps且达到历史极值区)，且【今日】实际利率开始回落(恐慌进入衰竭期)
        # 此时为极佳的左侧抄底点
        ry_zscore = (real_yield - real_yield.rolling(252).mean()) / real_yield.rolling(252).std()
        ry_diff1 = real_yield.diff(1)
        dash_for_cash_exhaustion = (ry_diff5 > 0.35) & (ry_zscore > 2.5) & (ry_diff1 < 0)
        
        # 赋值脉冲信号
        signal[dovish_easing] = 1.0
        signal[hawkish_tightening] = -1.0
        signal[dash_for_cash_exhaustion] = 1.0  # 衰竭信号具有最高优先级，覆盖为看多
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"