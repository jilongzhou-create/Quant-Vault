import numpy as np
import pandas as pd

class CrossAssetVolPanicExhaustionFactor:
    """跨资产波动率恐慌衰竭因子 (microstructure/options)

    逻辑: 计算标普500隐含波动率(VIX)与黄金隐含波动率(GVZ)的差值。当该差值极度飙升时, 意味着股市遭遇了远超传统避险资产(黄金)的极端流动性冲击(如2020年3月)。由于流动性冲击初期美债也可能遭到无差别抛售(金融机构被迫卖出高质量资产筹集美元现金), 因此绝对不能在VIX极值处盲目接飞刀。必须等待该波动率差值触顶并开始回落(跌破3日均值)时, 确认恐慌衰竭、流动性危机边际缓解, 此时资金将重新涌入高确定性的美债(TLT)避险, 这才是狙击手级别的抄底脉冲时机。
    数据: vixcls (CBOE VIX), gvzcls (CBOE 黄金ETF隐含波动率)
    触发: (VIX - GVZ) 的 252 日 Z-Score > 2.5 (恐慌极值) 且 当日差值 < 过去3日均值 (二阶导数/衰竭确认)
    输出: +1.0 (强烈看多美债脉冲), 常态为 0.0
    """

    def __init__(self, zscore_window: int = 252, zscore_threshold: float = 2.5, exhaust_window: int = 3):
        self.name = 'cross_asset_vol_panic_exhaustion'
        self.zscore_window = zscore_window
        self.zscore_threshold = zscore_threshold
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 初始信号全部为 0.0 (零值休眠)
        signal = pd.Series(0.0, index=data.index)

        # 检查数据完备性
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        # 处理缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 计算跨资产恐慌溢价 (股票隐含波动率 - 黄金隐含波动率)
        vol_spread = vix - gvz

        # 计算 252 个交易日 (一年) 的滚动 Z-Score
        rolling_mean = vol_spread.rolling(window=self.zscore_window).mean()
        rolling_std = vol_spread.rolling(window=self.zscore_window).std()
        
        # 避免除以 0 的情况
        z_score = (vol_spread - rolling_mean) / (rolling_std + 1e-8)

        # 计算用于判断边际衰竭的短期均值
        exhaustion_mean = vol_spread.rolling(window=self.exhaust_window).mean()

        # 铁律2: 二阶导数判断 (Anti-Catch-Falling-Knife)
        # 条件1: 跨资产恐慌溢价处于极端高位
        extreme_panic_condition = z_score > self.zscore_threshold
        
        # 条件2: 恐慌情绪边际减弱 (当前值低于3日均值，代表上涨动能衰竭并拐头)
        exhaustion_reversal_condition = vol_spread < exhaustion_mean

        # 只有在极端恐慌发生 且 开始衰竭的瞬间，才触发看多脉冲
        long_trigger = extreme_panic_condition & exhaustion_reversal_condition

        # 赋值触发脉冲
        signal.loc[long_trigger] = 1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, zscore_threshold={self.zscore_threshold}, exhaust_window={self.exhaust_window})"