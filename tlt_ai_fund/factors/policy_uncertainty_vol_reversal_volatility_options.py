import numpy as np
import pandas as pd

class PolicyUncertaintyVolReversalFactor:
    """政策不确定性与波动率衰竭共振 (volatility/options)

    逻辑: 政策不确定性(USEPUINDXD)和市场恐慌(VIX)的共振飙升意味着宏观悲观预期拉满，期限溢价上升打压长债。当两者从局部极端高位开始边际回落时，意味着政策"靴子落地"或恐慌消散，期限溢价骤降，催生美债强力做多脉冲。反之，极度平静被打破时产生做空脉冲。
    数据: usepuindxd, vixcls
    触发: 联合60日Z-Score > 1.5 且边际双双跌破3日均线触发 +1.0；联合Z-Score < -1.0 且边际双双向上突破3日均线触发 -1.0。
    输出: +1.0看多美债，-1.0看空美债，常态严格为 0.0 (脉冲型)。
    """

    def __init__(self):
        self.name = 'policy_uncertainty_vol_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠, 初始信号严格全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['usepuindxd', 'vixcls']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 数据预处理: 填充缺失值并防前向数据泄露
        df = data[req_cols].ffill()
        
        epu = df['usepuindxd']
        vix = df['vixcls']
        
        # 计算 60 日滚动 Z-Score 确定极端极值状态
        window = 60
        
        epu_mean = epu.rolling(window).mean()
        epu_std = epu.rolling(window).std().replace(0, np.nan)
        epu_z = (epu - epu_mean) / epu_std
        
        vix_mean = vix.rolling(window).mean()
        vix_std = vix.rolling(window).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        # 联合恐慌得分 (平滑两者的异步杂音)
        combo_z = (epu_z + vix_z) / 2.0
        
        # 铁律3: 边际变化 (使用 3 日均值作为脉冲突破基准)
        epu_ma3 = epu.rolling(3).mean()
        vix_ma3 = vix.rolling(3).mean()
        
        # 铁律2: 二阶导数 (极值 + 开始回落) -> 防止接飞刀
        # 多头触发: 极度恐慌且不确定性高企，同时两者动能瓦解
        long_cond = (
            (combo_z > 1.5) &             # 条件1: 联合恐慌处于局部高位极值
            (combo_z.diff() < 0) &        # 条件2: 联合指标边际回落
            (epu < epu_ma3) &             # 条件3: EPU动量跌破3日均线 (靴子落地)
            (vix < vix_ma3)               # 条件4: VIX动量跌破3日均线 (恐慌消散)
        )
        
        # 空头触发: 极度平静与自满，被突发变化打破
        short_cond = (
            (combo_z < -1.0) &            # 条件1: 联合恐慌处于局部极度平静状态
            (combo_z.diff() > 0) &        # 条件2: 波动开始复苏
            (epu > epu_ma3) &             # 条件3: EPU向上突破3日均线
            (vix > vix_ma3)               # 条件4: VIX向上突破3日均线
        )
        
        # 仅在触发日赋予脉冲非零值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"