import numpy as np
import pandas as pd

class UnstructuredOptionsPolicyPanicFactor:
    """Unstructured Options Policy Panic Factor (unstructured/options)

    逻辑: 结合非结构化的经济政策不确定性指数(EPU)脉冲与黄金期权隐含波动率(GVZ)捕捉避险资金流向。当政策不确定性边际飙升引发跨资产恐慌，且期权市场恐慌见顶衰竭时，代表流动性抛售结束，避险资金将确定性买入美债(TLT)锁定收益；反之则抛售美债拥抱风险资产。
    数据: usepuindxd (经济政策不确定性), gvzcls (黄金ETF隐含波动率)
    触发: 
      看多: EPU 5日变化量 Z-Score > 2.5 AND GVZ Z-Score > 2.5 AND GVZ < GVZ 3日均值 (恐慌衰竭)
      看空: EPU 5日变化量 Z-Score < -2.5 AND GVZ Z-Score < -2.0 AND GVZ > GVZ 3日均值 (安逸破灭)
    输出: 脉冲信号 [-1.0, 0.0, +1.0]
    """

    def __init__(self, window=252, diff_days=5, smooth_days=3, z_th_up=2.5, z_th_dn=-2.5, gvz_th_up=2.5, gvz_th_dn=-2.0):
        self.name = 'unstructured_options_policy_panic'
        self.window = window
        self.diff_days = diff_days
        self.smooth_days = smooth_days
        self.z_th_up = z_th_up
        self.z_th_dn = z_th_dn
        self.gvz_th_up = gvz_th_up
        self.gvz_th_dn = gvz_th_dn

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 验证必需的数据字段是否存在
        if 'usepuindxd' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 (对非结构化的政策不确定性数据求边际变化量，绝对禁止直接用绝对水位)
        epu_diff = epu.diff(self.diff_days)
        
        # 计算政策不确定性边际突变的 Z-Score
        epu_diff_mean = epu_diff.rolling(self.window).mean()
        epu_diff_std = epu_diff.rolling(self.window).std()
        epu_z = (epu_diff - epu_diff_mean) / (epu_diff_std + 1e-8)
        
        # 计算黄金期权隐含波动率的 Z-Score
        gvz_mean = gvz.rolling(self.window).mean()
        gvz_std = gvz.rolling(self.window).std()
        gvz_z = (gvz - gvz_mean) / (gvz_std + 1e-8)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife，期权波动率必须出现拐点)
        gvz_ma3 = gvz.rolling(self.smooth_days).mean()
        gvz_exhaustion_bull = gvz < gvz_ma3  # 波动率见顶回落，无脑抛售引发的流动性危机解除
        gvz_exhaustion_bear = gvz > gvz_ma3  # 波动率触底反弹，市场重燃动物精神
        
        # 铁律1: 零值休眠 (狙击手脉冲)
        bull_cond = (epu_z > self.z_th_up) & (gvz_z > self.gvz_th_up) & gvz_exhaustion_bull
        bear_cond = (epu_z < self.z_th_dn) & (gvz_z < self.gvz_th_dn) & gvz_exhaustion_bear
        
        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, diff_days={self.diff_days})"