import numpy as np
import pandas as pd

class GoldVixRatioExhaustionOptionsFactor:
    """黄金与股市波动率背离衰竭因子 (unstructured/options)

    逻辑: 黄金隐含波动率(GVZ)与股市隐含波动率(VIX)的比值能够区分宏观冲击的性质。当VIX极端主导导致比率崩盘时，意味着纯粹的系统性流动性危机(如20年3月)，这种危机必将逼迫美联储超级放水，在恐慌拐点做多美债(TLT)。反之，当GVZ主导飙升时，暗示市场担忧恶性通胀或地缘失控，此时衰竭后会推升长端利率，做空美债。此策略为脉冲狙击手型，常态静默。
    数据: gvzcls, vixcls
    触发: 比率的 252日 Z-Score 达极值 (Z > 2.5 或 Z < -2.5) AND 伴随短期均值反转(3日衰竭)
    输出: +1.0 (流动性恐慌衰竭, 放水预期看多TLT), -1.0 (滞胀恐慌衰竭, 利率重估看空TLT)
    """

    def __init__(self, z_threshold=2.5, window=252, exhaust_win=3):
        self.name = 'gold_vix_ratio_exhaustion'
        self.z_threshold = z_threshold
        self.window = window
        self.exhaust_win = exhaust_win

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态返回0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 校验所需数据是否存在
        if 'gvzcls' not in data.columns or 'vixcls' not in data.columns:
            return signal

        # 提取并填充缺失值
        gvz = data['gvzcls'].ffill()
        vix = data['vixcls'].ffill()
        
        # 避免除以 0 以及无穷大
        vix = vix.replace(0, np.nan).ffill()
        ratio = gvz / vix
        
        # 计算历史滚动极值判定: 252日(约1年)滚动 Z-Score
        roll_mean = ratio.rolling(window=self.window, min_periods=60).mean()
        roll_std = ratio.rolling(window=self.window, min_periods=60).std()
        z_score = (ratio - roll_mean) / (roll_std + 1e-6)
        
        # 铁律2: 二阶导数(衰竭条件)，计算近期3日均线用于判定反转
        short_ma = ratio.rolling(window=self.exhaust_win).mean()
        
        # 触发 1: 流动性恐慌主导 (VIX极高 -> ratio极低), 且开始反转回升
        # 逻辑: 极度恐慌见顶 -> 美联储被迫宽松预期开启 -> 爆买长端美债 (+1.0)
        long_cond = (z_score < -self.z_threshold) & (ratio > short_ma)
        
        # 触发 2: 滞胀/地缘恐慌主导 (GVZ极高 -> ratio极高), 且开始反转回落
        # 逻辑: 特殊恐慌消退 -> 避险情绪撤出且通胀预期推高长端利率 -> 抛售美债 (-1.0)
        short_cond = (z_score > self.z_threshold) & (ratio < short_ma)
        
        # 赋值并只保留满足条件的狙击脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_threshold={self.z_threshold}, window={self.window}, exhaust_win={self.exhaust_win})"