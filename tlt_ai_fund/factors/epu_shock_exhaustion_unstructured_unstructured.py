import numpy as np
import pandas as pd

class EpuMomentumShockFactor:
    """Epu Momentum Shock (unstructured/unstructured)

    逻辑: 采用基于日常新闻文本提取的美国经济政策不确定性指数(usepuindxd)，衡量宏观政策层面的恐慌与自满。
          该因子完全放弃FOMC货币政策的文本得分，转而从广义新闻的非结构化情绪(政策不确定性)入手，捕捉避险情绪极值。
          当不确定性极大飙升(Z-Score > 2.0)且开始回落时，说明避险情绪见顶，资金重新流向风险资产，看空美债；
          当市场极度自满(Z-Score < -2.0)且不确定性突然反转上升时，说明突发黑天鹅导致避险资金涌入，看多美债。
    数据: usepuindxd (Daily News-based US Economic Policy Uncertainty Index)
    触发: 10日与60日均值的边际动量，其252日Z-Score达到极端(>2.0 或 <-2.0)，且该动量跌破/升破其3日均值(衰竭拐点)。
    输出: [-1.0, 1.0] 的避险反转脉冲信号。
    """

    def __init__(self, short_window=10, long_window=60, z_window=252, threshold=2.0):
        self.name = 'epu_momentum_shock'
        self.short_window = short_window
        self.long_window = long_window
        self.z_window = z_window
        self.threshold = threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 初始信号必须为0.0 (零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 提取基于非结构化新闻文本的经济政策不确定性，并处理缺失值
        epu = data['usepuindxd'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change)
        # 绝对禁止直接判断 EPU 水位！我们计算 2周(10日) 相对于 1个季度(60日) 的不确定性爆发动量
        epu_short = epu.rolling(window=self.short_window, min_periods=1).mean()
        epu_long = epu.rolling(window=self.long_window, min_periods=1).mean()
        epu_mom = epu_short - epu_long
        
        # 计算1年期(252交易日)的滚动Z-Score，衡量当前爆发或骤降的极端程度
        mom_mean = epu_mom.rolling(window=self.z_window, min_periods=self.z_window//2).mean()
        mom_std = epu_mom.rolling(window=self.z_window, min_periods=self.z_window//2).std()
        
        zscore = (epu_mom - mom_mean) / (mom_std + 1e-6)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 计算动量的3日均线，作为二阶衰竭的判定基准
        epu_mom_ma3 = epu_mom.rolling(window=3, min_periods=1).mean()
        
        # 卖出脉冲(做空美债): 极端政策恐慌(Z-Score > 2.0)且开始见顶消退(动量跌破3日均值) -> 资金流出避险资产
        short_cond = (zscore > self.threshold) & (epu_mom < epu_mom_ma3)
        
        # 买入脉冲(做多美债): 极度政策自满(Z-Score < -2.0)且突然抬头恶化(动量升破3日均值) -> 资金恐慌涌入避险资产
        long_cond = (zscore < -self.threshold) & (epu_mom > epu_mom_ma3)
        
        # 赋值脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"EpuMomentumShockFactor(short_window={self.short_window}, long_window={self.long_window}, z_window={self.z_window}, threshold={self.threshold})"