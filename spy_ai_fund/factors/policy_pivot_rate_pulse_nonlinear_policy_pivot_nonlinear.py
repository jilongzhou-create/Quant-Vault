import numpy as np
import pandas as pd

class PolicyPivotRatePulseNonlinearFactor:
    """政策转向利率冲量因子 (policy_pivot/nonlinear)

    逻辑: 捕捉市场对美联储政策预期的极短期剧烈重定价。当2年期美债收益率在3天内暴跌且曲线急剧变陡，同时高收益信用利差未大幅恶化时，过滤了单纯避险情绪，确认流动性实质宽松预期，发出看多脉冲；反之，若短端利率飙升、曲线倒挂加深且信用利差走阔，确认紧缩预期与戴维斯双杀恶化，发出看空脉冲。
    数据: dgs2, t10y2y, bamlh0a0hym2
    输出: 政策宽松确认看多为+1.0，紧缩恶化看空为-1.0，常态下为0.0
    触发条件: DGS2 3日变动绝对值超过15bps，且T10Y2Y变动幅度超过8bps，配合信用利差过滤，预期 Trigger Rate 5%-15%
    """

    def __init__(self):
        self.name = 'policy_pivot_rate_pulse_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查所需数据列是否存在
        req_cols = ['dgs2', 't10y2y', 'bamlh0a0hym2']
        if not all(col in data.columns for col in req_cols):
            return pd.Series(0.0, index=data.index, name=self.name)

        # 前向填充缺失值以应对节假日/停牌
        df = data[req_cols].ffill()
        
        # 核心原则：关注动量变化而不是绝对水位 (边际变化铁律)
        # 计算极短窗口(3个交易日)内的基点剧烈变化
        dgs2_3d = df['dgs2'].diff(3)
        t10y2y_3d = df['t10y2y'].diff(3)
        cred_3d = df['bamlh0a0hym2'].diff(3)

        # 多头条件：市场抢跑实质性降息(Bull Steepening) 且 未发生信用危机
        # 1. 短端利率暴跌超15个基点
        # 2. 收益率曲线急剧变陡超8个基点
        # 3. 信用利差没有飙升(涨幅<=5个基点)，说明不是由于突发危机导致的避险(Flight to safety)，而是实质的政策转鸽
        bull_cond = (dgs2_3d < -0.15) & (t10y2y_3d > 0.08) & (cred_3d <= 0.05)
        
        # 空头条件：市场预期紧缩加剧(Bear Flattening) 且 信用环境恶化
        # 1. 短端利率飙升超15个基点(杀估值)
        # 2. 收益率曲线急剧变平或倒挂加深超8个基点
        # 3. 信用利差走阔(涨幅>5个基点)，流动性同时收紧
        bear_cond = (dgs2_3d > 0.15) & (t10y2y_3d < -0.08) & (cred_3d > 0.05)

        # 零值休眠铁律：常态下必须返回0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 极端事件瞬间触发脉冲
        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"