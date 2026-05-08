import numpy as np
import pandas as pd

class GvzPanicExhaustionFactor:
    """黄金波动率恐慌极值与衰竭反转因子 (microstructure/unstructured)

    逻辑: 黄金与美债同属避险资产，当黄金波动率(GVZ)达到极端高位时，表明宏观层面出现流动性危机，避险资产遭遇无差别抛售。只有当 GVZ 触及极端高位并开始见顶回落（恐慌衰竭）时，才意味着流动性冲击解除，避险资产将迎来报复性反弹。此因子专为捕捉该类脉冲机会而设计，平时保持休眠。
    数据: gvzcls (黄金波动率指数)
    触发: GVZ 252日 Z-Score > 2.5 (恐慌极值) 且 当日 GVZ < 过去3日均值 (恐慌衰竭/回落)
    输出: +1.0 (流动性危机解除，看多美债脉冲)，非触发日严格为 0.0
    """

    def __init__(self, zscore_window=252, threshold=2.5, exhaust_window=3):
        self.name = 'gvz_panic_exhaustion'
        self.zscore_window = zscore_window
        self.threshold = threshold
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全零 Series，严格遵守零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        if 'gvzcls' not in data.columns:
            return signal

        # 提取数据并处理缺失值
        gvz = data['gvzcls'].ffill()
        
        # 计算 252交易日 (一年) 的滚动 Z-Score 以衡量极值
        roll_mean = gvz.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).mean()
        roll_std = gvz.rolling(window=self.zscore_window, min_periods=self.zscore_window//2).std()
        
        # 防止除零异常
        roll_std = roll_std.replace(0.0, np.nan)
        zscore = (gvz - roll_mean) / roll_std
        
        # 衰竭条件 (二阶导数铁律): 波动率必须开始回落，禁止直接在左侧飙升期接飞刀
        exhaust_mean = gvz.rolling(window=self.exhaust_window).mean()
        exhaust_condition = gvz < exhaust_mean
        
        # 触发看多脉冲：恐慌达到极值 且 恐慌开始衰竭
        buy_pulse = (zscore > self.threshold) & exhaust_condition
        
        # 仅在条件满足的瞬间赋 +1.0，其余时间保持默认的 0.0
        signal.loc[buy_pulse] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, threshold={self.threshold}, exhaust_window={self.exhaust_window})"