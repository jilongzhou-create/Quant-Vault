import numpy as np
import pandas as pd

class UnstructuredVolShockReversalFactor:
    """UnstructuredVolShockReversal (volatility/unstructured)

    逻辑: 结合跨资产波动率极值与非结构化数据(NLP新闻/FOMC情绪)的突变衰竭。
          当市场处于极端政策恐慌(EPU或VIX的年度Z-Score>2.5)且跨资产全面确认二阶导数衰竭(跌破3日均值)时，
          或者当FOMC文本情绪发生罕见的鹰转鸽极端突变时，由于紧缩预期瓦解，输出看多美债脉冲；
          反之，当FOMC情绪发生极端的鸽转鹰突变时，输出看空美债脉冲。
    数据: vixcls (VIX), gvzcls (黄金波动率), usepuindxd (经济政策不确定性), fomc_sentiment (FOMC情绪得分)
    触发: 
      看多(+1.0): (VIX或EPU 252日Z-Score>2.5 且 VIX/GVZ/EPU同步回落跌破均线) OR (FOMC情绪5日变化Z>2.5且由负转正)
      看空(-1.0): (FOMC情绪5日变化Z < -2.5且由正转负)
    输出: [-1.0, 1.0] 的狙击手级脉冲信号
    """

    def __init__(self):
        self.name = 'unstructured_vol_shock_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        long_trigger = pd.Series(False, index=data.index)
        short_trigger = pd.Series(False, index=data.index)
        
        # === 核心逻辑 1: 极端波动率与政策恐慌衰竭 (看多) ===
        # 必须同时满足极值与衰竭条件 (铁律2: Anti-Catch-Falling-Knife)
        req_vol = ['vixcls', 'gvzcls', 'usepuindxd']
        if all(c in data.columns for c in req_vol):
            vix = data['vixcls'].ffill()
            gvz = data['gvzcls'].ffill()
            epu = data['usepuindxd'].ffill() # 每日经济政策不确定性 (新闻非结构化转化)
            
            # 铁律3: 计算 252日(一年) Z-Score 以识别宏观极端拥挤水位
            vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std().replace(0, 1e-6)
            epu_z = (epu - epu.rolling(252).mean()) / epu.rolling(252).std().replace(0, 1e-6)
            
            # 条件1: 市场波动率或政策不确定性处于极端恐慌状态
            extreme_panic = (vix_z > 2.5) | (epu_z > 2.5)
            
            # 条件2: VIX 二阶导数衰竭 (差值为负且跌破3日短期均线)
            vix_exhausted = (vix.diff() < 0) & (vix < vix.rolling(3).mean())
            
            # 条件3: 跨资产确认 (黄金避险波动率与非结构化政策不确定性同步边际回落)
            cross_asset_exhausted = (gvz.diff() < 0) & (epu.diff() < 0)
            
            vol_long = extreme_panic & vix_exhausted & cross_asset_exhausted
            long_trigger = long_trigger | vol_long

        # === 核心逻辑 2: 非结构化 FOMC 文本情绪突变 (看多与看空) ===
        # 铁律3: 绝对禁止直接使用阶梯数据的绝对值，必须使用滚动变化量
        if 'fomc_sentiment' in data.columns:
            fomc = data['fomc_sentiment'].ffill()
            
            # 计算5日(单周)变化量以捕捉FOMC会议前后的突变跳跃
            fomc_diff = fomc.diff(5)
            fomc_diff_z = (fomc_diff - fomc_diff.rolling(252).mean()) / fomc_diff.rolling(252).std().replace(0, 1e-6)
            
            # 鹰转鸽反转 (Dovish Shock) -> 紧缩恐慌解除，强力利好美债
            hawk_to_dove = (fomc.shift(5) < 0) & (fomc > 0)
            dovish_shock = (fomc_diff_z > 2.5) & hawk_to_dove
            long_trigger = long_trigger | dovish_shock
            
            # 鸽转鹰反转 (Hawkish Shock) -> 宽松预期破灭，强力利空美债
            dove_to_hawk = (fomc.shift(5) > 0) & (fomc < 0)
            hawkish_shock = (fomc_diff_z < -2.5) & dove_to_hawk
            short_trigger = short_trigger | hawkish_shock

        # 铁律1: 零值休眠 (平时为0.0，只在触发日输出脉冲)
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0
        
        signal.name = self.name
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"