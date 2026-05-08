import numpy as np
import pandas as pd

class PolicyUncertaintyReversalFactor:
    """政策不确定性极端反转脉冲因子 (volatility/unstructured)

    逻辑: 经济政策不确定性(USEPU,基于非结构化新闻文本计算)经历极端飙升后, 若开始回落且黄金避险波动率(GVZ)同步衰竭, 标志着宏观恐慌挤兑见顶, 资金将重新流入具备确定性的美债(TLT), 触发看多脉冲。反之, 极度自满后的反弹则看空。
    数据: usepuindxd (经济政策不确定性指数), gvzcls (黄金波动率指数)
    触发: USEPU 5日变化量的 252日 Z-Score > 2.5(极值) + USEPU及GVZ单日diff<0且低于3日均值(衰竭确认)
    输出: 狙击手级脉冲信号, 范围 [-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'policy_uncertainty_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (默认全 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 处理数据缺失情况
        if 'usepuindxd' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 (绝对禁止使用原值, 使用 5个交易日的变化量捕捉预期突变)
        epu_chg = epu.diff(5)
        
        # 计算一年期 (252日) 的滚动 Z-Score 衡量事件极端程度
        epu_chg_mean = epu_chg.rolling(window=252, min_periods=126).mean()
        epu_chg_std = epu_chg.rolling(window=252, min_periods=126).std()
        epu_chg_z = (epu_chg - epu_chg_mean) / epu_chg_std.replace(0, np.nan)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife, 必须等极值开始衰竭才能交易)
        # 看多衰竭条件: 政策恐慌回落 且 黄金波动率同步降温
        epu_falling = (epu.diff(1) < 0) & (epu < epu.rolling(window=3).mean())
        gvz_falling = (gvz.diff(1) < 0) & (gvz < gvz.rolling(window=3).mean())
        
        # 看空衰竭条件: 市场极度自满后, 恐慌与波动率重新抬头
        epu_rising = (epu.diff(1) > 0) & (epu > epu.rolling(window=3).mean())
        gvz_rising = (gvz.diff(1) > 0) & (gvz > gvz.rolling(window=3).mean())
        
        # 信号触发逻辑组合
        # 多头脉冲 (+1.0): 恐慌边际狂飙(极值) + 跨资产恐慌同步回落(衰竭)
        long_cond = (epu_chg_z > 2.5) & epu_falling & gvz_falling
        
        # 空头脉冲 (-1.0): 恐慌罕见骤降(极负值) + 跨资产恐慌死灰复燃(反弹)
        short_cond = (epu_chg_z < -2.5) & epu_rising & gvz_rising
        
        # 赋值触发脉冲 (非触发日严格保持为0)
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"