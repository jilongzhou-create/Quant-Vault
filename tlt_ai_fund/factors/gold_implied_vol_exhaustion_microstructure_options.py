import numpy as np
import pandas as pd

class GoldImpliedVolExhaustionFactor:
    """黄金隐含波动率极值衰竭因子 (microstructure/options)

    逻辑: 黄金(极佳避险与抗通胀资产)的隐含波动率(GVZCLS)微观结构能前瞻性反映宏观状态的切换。当其极端飙升时, 意味着发生了流动性危机(连黄金都被无差别抛售以换取美元), 而当其见顶回落时表明美联储救助生效、恐慌衰竭, 是绝佳的看多美债(TLT)时机; 反之, 当黄金波动率极度低迷且开始抬头时, 往往暗示通胀预期或尾部风险重新苏醒, 导致紧缩预期升温, 触发脉冲看空美债信号。因子严格设计为脉冲形态。
    数据: gvzcls (CBOE黄金ETF隐含波动率)
    触发: 
      多头: gvzcls 的 252日 Z-Score > 2.5 且 gvzcls < 3日均值 (恐慌高位衰竭)
      空头: gvzcls 的 252日 Z-Score < -2.0 且 gvzcls > 3日均值 (极度自满后觉醒)
    输出: +1.0 看多美债脉冲, -1.0 看空美债脉冲, 否则为 0.0
    """

    def __init__(self, zscore_window: int = 252, zscore_long: float = 2.5, zscore_short: float = -2.0, exhaust_window: int = 3):
        self.name = 'gold_implied_vol_exhaustion'
        self.zscore_window = zscore_window
        self.zscore_long = zscore_long
        self.zscore_short = zscore_short
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠, 初始信号严格全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 校验必备数据缺失
        if 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充缺失值以保持日频连续性
        gvz = data['gvzcls'].ffill()
        
        # 计算统计学基准极值 (252个交易日具有一年的经济学周期含义)
        roll_mean = gvz.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).mean()
        roll_std = gvz.rolling(window=self.zscore_window, min_periods=self.zscore_window // 2).std()
        zscore = (gvz - roll_mean) / (roll_std + 1e-8)
        
        # 铁律3: 边际变化 (二阶导数观察点), 以极短期均值代表即期趋势水位
        short_term_mean = gvz.rolling(window=self.exhaust_window).mean()
        
        # 铁律2: 二阶导数 (必须包含极值条件 + 衰竭/反转条件, 绝不接飞刀!)
        # 极度恐慌 + 恐慌开始消退 -> 看多美债
        buy_cond = (zscore > self.zscore_long) & (gvz < short_term_mean)
        
        # 极度自满 + 波动率/风险重新觉醒 -> 看空美债
        sell_cond = (zscore < self.zscore_short) & (gvz > short_term_mean)
        
        # 输出脉冲信号
        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"GoldImpliedVolExhaustionFactor(zscore_window={self.zscore_window}, zscore_long={self.zscore_long}, zscore_short={self.zscore_short})"