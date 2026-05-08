import numpy as np
import pandas as pd

class CreditQualityPanicExhaustionFactor:
    """信用质量恐慌衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 衡量高收益债(HY)与BBB级投资债(BBB)的信用利差差值(Junk Premium)。垃圾债溢价代表了市场对企业违约的纯粹恐慌。
          当该溢价极度走阔后出现收窄(恐慌见顶衰竭)，发出强烈抄底做多美股信号；当溢价温和但持续走阔时(钝刀割肉期)，发出看空美股信号。
    数据: bamlh0a0hym2 (高收益债OAS), bamlc0a4cbbb (BBB级债OAS)
    输出: 脉冲型信号。极度恐慌且开始收窄时输出+1.0，温和恐慌且恶化时输出-1.0，常态输出0.0
    触发条件: 垃圾债溢价Z-Score > 1.5 且 3日变化率 < 0 触发看多；0.5 < Z-Score <= 1.5 且 3日变化率 > 0 触发看空。预期Trigger Rate ~12%
    """

    def __init__(self):
        self.name = 'credit_quality_panic_exhaustion_pulse'
        self.zscore_window = 252
        self.momentum_window = 3
        self.extreme_z_threshold = 1.5
        self.mild_z_threshold = 0.5

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        required_cols = ['bamlh0a0hym2', 'bamlc0a4cbbb']
        
        # 检查所需数据列是否存在
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0.0, index=data.index, name=self.name)
            
        hy_spread = data['bamlh0a0hym2'].ffill()
        bbb_spread = data['bamlc0a4cbbb'].ffill()
        
        # 计算 Junk Premium (高收益债溢价: 剔除了整体基准利率和基础投资级风险后的纯粹垃圾债违约恐慌)
        junk_premium = hy_spread - bbb_spread
        
        # 计算 252日 Z-Score (代表市场长期基准下的相对偏离度)
        rolling_mean = junk_premium.rolling(window=self.zscore_window, min_periods=60).mean()
        rolling_std = junk_premium.rolling(window=self.zscore_window, min_periods=60).std()
        z_score = (junk_premium - rolling_mean) / (rolling_std + 1e-6)
        
        # 计算动量变化 (3日边际变化，识别趋势是否开始反转)
        premium_diff = junk_premium.diff(self.momentum_window)
        
        # 初始化脉冲信号 (零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 买入条件 (防接飞刀法则)：信用恐慌达到极值 (Z > 1.5) 且 恐慌开始衰竭/利差收窄 (diff < 0)
        buy_cond = (z_score > self.extreme_z_threshold) & (premium_diff < 0)
        
        # 卖出条件：信用恐慌处于温和恶化状态 (0.5 < Z <= 1.5) 且 还在持续走阔 (diff > 0) -> 典型的钝刀割肉主跌浪阶段
        sell_cond = (z_score > self.mild_z_threshold) & (z_score <= self.extreme_z_threshold) & (premium_diff > 0)
        
        # 赋值脉冲信号
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        # 处理可能由于缺失值导致的 NaN，确保完全遵循常态 0.0 铁律
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, momentum_window={self.momentum_window})"