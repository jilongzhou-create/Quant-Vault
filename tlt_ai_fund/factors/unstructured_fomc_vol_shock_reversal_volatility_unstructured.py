import numpy as np
import pandas as pd

class UnstructuredFomcVolShockReversalFactor:
    """Unstructured FOMC Volatility Shock Reversal Factor (volatility/unstructured)

    逻辑: 将NLP提取的FOMC鹰鸽情绪边际突变与跨资产恐慌波动率(VIX)结合。加息恐慌中，若美联储边际意外转鸽且恐慌情绪见顶回落，标志流动性冲击瓦解，触发抄底美债脉冲。意外转鹰且波动率跳升则触发做空脉冲。
    数据: fomc_sentiment (非结构化情绪), vixcls (恐慌波动率)
    触发: 多头 -> FOMC情绪鸽派突变(Z-Score > 2.5) AND VIX处于相对高位(Z-Score > 1.5) AND 波动率衰竭(diff < 0 且低于3日均值)
    输出: [-1.0, 1.0] 仅在预期突变及波动率反转瞬间触发的零值休眠脉冲
    """

    def __init__(self):
        self.name = 'unstructured_fomc_vol_shock_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态输出 0.0
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖列是否存在
        required_cols = ['fomc_sentiment', 'vixcls']
        if not all(col in data.columns for col in required_cols):
            return signal

        fomc = data['fomc_sentiment'].ffill()
        vix = data['vixcls'].ffill()

        # 铁律3: 边际变化 (Marginal Change Only)
        # 严禁直接使用 fomc_sentiment 绝对值！提取超预期预期的跳点动量
        fomc_diff = fomc.diff().fillna(0.0)
        fomc_diff_mean = fomc_diff.rolling(window=252, min_periods=21).mean()
        fomc_diff_std = fomc_diff.rolling(window=252, min_periods=21).std().replace(0.0, np.nan)
        
        # FOMC情绪动量 Z-Score，用于捕捉极端鸽派/鹰派意外突变
        fomc_z = (fomc_diff - fomc_diff_mean) / fomc_diff_std
        fomc_z = fomc_z.fillna(0.0)

        # 波动率水位衡量 (用于防止平静期接飞刀)
        vix_mean = vix.rolling(window=252, min_periods=21).mean()
        vix_std = vix.rolling(window=252, min_periods=21).std().replace(0.0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        # 波动率变化与移动均值
        vix_diff = vix.diff()
        vix_rolling_3m = vix.rolling(window=3, min_periods=1).mean()

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 条件A: VIX处于前期恐慌累积的高位 (但不一定要破2.5那么极寒，因为这需要配合FOMC事件)
        vix_high = vix_z > 1.5 
        # 条件B: 恐慌必须出现衰竭，不能继续飙升，停止接飞刀
        vix_exhaustion = (vix_diff < 0) & (vix < vix_rolling_3m)

        # 多头脉冲 (Sniper Pulse): 3天内有重磅鸽派信号突变，同时当前VIX从高位确认回落
        dove_shock = fomc_z > 2.5
        dove_shock_recent = dove_shock.rolling(window=3, min_periods=1).max() > 0
        long_cond = dove_shock_recent & vix_high & vix_exhaustion

        # 空头脉冲: 3天内有重磅鹰派信号突变，同时波动率抬头飙升，刺破平静期
        hawk_shock = fomc_z < -2.5
        hawk_shock_recent = hawk_shock.rolling(window=3, min_periods=1).max() > 0
        vix_surging = (vix_diff > 0) & (vix > vix_rolling_3m)
        short_cond = hawk_shock_recent & vix_surging

        # 严格赋值，保证其余时间为休眠 0.0
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"