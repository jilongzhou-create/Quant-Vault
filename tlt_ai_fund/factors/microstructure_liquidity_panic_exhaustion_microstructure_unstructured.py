import numpy as np
import pandas as pd

class MicrostructureLiquidityPanicExhaustionFactor:
    """微观流动性恐慌衰竭脉冲因子 (microstructure/unstructured)

    逻辑: 芝加哥联储国家金融状况指数(NFCI)降维提炼了上百种微观结构与流动性数据。当NFCI的短期恶化动量创极端极值时，意味着市场微观结构破裂，发生类似2020年3月的美元荒与无差别抛售(美债遭流动性错杀)。当指数攀升见顶并开始边际回落(衰竭)时，标志着联储注水起效或流动性挤兑结束，避险资金将报复性回补并做多美债(TLT)。
    数据: nfci (芝加哥联储国家金融状况指数)
    触发: NFCI 5日变化量的 252日 Z-Score > 2.5 (流动性挤兑突变) AND 当日 NFCI < 过去3日均值 (流动性恶化衰竭回落)
    输出: +1.0 脉冲 (看多美债)
    """

    def __init__(self, zscore_window=252, diff_window=5, zscore_threshold=2.5, exhaust_window=3):
        self.name = 'microstructure_liquidity_panic_exhaustion'
        self.zscore_window = zscore_window
        self.diff_window = diff_window
        self.zscore_threshold = zscore_threshold
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始信号全为0.0，只在极端事件触发脉冲
        signal = pd.Series(0.0, index=data.index)
        
        # 处理缺少数据列的鲁棒性
        if 'nfci' not in data.columns:
            return signal
            
        # 前向填充缺失值，防止 NaN 干扰计算
        nfci = data['nfci'].ffill()
        
        # 铁律3: 边际变化。绝对禁止直接输出低频数据的绝对值！
        # 使用短期变化量(动量)来捕捉微观流动性突发断裂的瞬间
        nfci_momentum = nfci.diff(self.diff_window)
        
        # 计算动量的滚动 Z-Score
        roll_mean = nfci_momentum.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        roll_std = nfci_momentum.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        
        # 防止除以0出现无穷大
        roll_std = roll_std.replace(0.0, np.nan)
        zscore = (nfci_momentum - roll_mean) / roll_std
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 条件1: 极值 - 微观流动性短期恶化动量达到极端高位
        extreme_panic = zscore > self.zscore_threshold
        
        # 条件2: 衰竭 - 当日压力指数开始下穿近期均值，代表最恐慌的单边恶化已经衰竭见顶
        exhaustion = nfci < nfci.rolling(window=self.exhaust_window).mean()
        
        # 只有极值和衰竭同时发生时才触发做多抄底脉冲
        trigger = extreme_panic & exhaustion
        
        signal[trigger] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"MicrostructureLiquidityPanicExhaustionFactor(zscore_window={self.zscore_window}, diff_window={self.diff_window}, zscore_threshold={self.zscore_threshold}, exhaust_window={self.exhaust_window})"