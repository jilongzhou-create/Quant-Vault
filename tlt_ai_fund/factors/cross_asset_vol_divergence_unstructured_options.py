import numpy as np
import pandas as pd

class CrossAssetVolDivergenceFactor:
    """跨资产波动率背离衰竭因子 (unstructured/options)

    逻辑: 捕捉美股恐慌(VIX)与黄金避险恐慌(GVZ)隐含波动率的极端背离。黄金与美债同具避险属性，当 VIX 飙升且远超 GVZ 时表明股市单边恐慌，
          由于“边际变化”与“二阶导数”铁律，因子只在波动率差值极速飙升（Z-Score > 2.5）且差值开始回落（<3日均线）时触发做多美债信号，
          此时股市恐慌衰退，避险资金涌入或美联储降息预期升温推动美债上行。反之，极度自满结束时做空。
    数据: vixcls (标普500隐含波动率), gvzcls (黄金ETF隐含波动率)
    触发: 波动率差值的 5日动量 Z-Score > 2.5 且 差值 < 3日均值 且 差值回落 -> +1.0
          波动率差值的 5日动量 Z-Score < -2.5 且 差值 > 3日均值 且 差值反弹 -> -1.0
    输出: 严格脉冲型信号，[-1.0, 1.0]
    """

    def __init__(self, z_threshold: float = 2.5, window: int = 252, diff_days: int = 5):
        self.name = 'cross_asset_vol_divergence'
        self.z_threshold = z_threshold
        self.window = window
        self.diff_days = diff_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 序列 (铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查数据列是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        # 填充缺失值，避免假期等因素导致 NaN
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 跨资产隐含波动率差值 (美股风险 - 黄金避险)
        vol_spread = vix - gvz

        # 铁律3: 边际变化（计算差值的短期动量，捕捉恐慌预期的跳跃瞬间）
        spread_diff = vol_spread.diff(self.diff_days)

        # 滚动 Z-Score 评估极端脉冲
        roll_mean = spread_diff.rolling(window=self.window, min_periods=60).mean()
        roll_std = spread_diff.rolling(window=self.window, min_periods=60).std()
        
        # 避免除以 0 导致 inf
        zscore = (spread_diff - roll_mean) / (roll_std + 1e-6)

        # 衰竭条件基准线
        spread_ma3 = vol_spread.rolling(window=3).mean()

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 做多触发：恐慌突变达到极值 (Z-Score > 2.5) AND 开始衰竭 (差值下穿3日均线 且 今日边际回落)
        long_cond = (
            (zscore > self.z_threshold) & 
            (vol_spread < spread_ma3) & 
            (vol_spread.diff(1) < 0)
        )

        # 做空触发：极度自满达到极值 (Z-Score < -2.5) AND 自满开始逆转 (差值上穿3日均线 且 今日边际反弹)
        short_cond = (
            (zscore < -self.z_threshold) & 
            (vol_spread > spread_ma3) & 
            (vol_spread.diff(1) > 0)
        )

        # 赋值极端脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_threshold={self.z_threshold}, window={self.window})"