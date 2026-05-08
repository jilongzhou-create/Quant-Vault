import numpy as np
import pandas as pd

class GoldVolatilityShockExhaustionFactor:
    """黄金期权波动率脉冲衰竭因子 (unstructured/options)

    逻辑: 捕捉黄金期权隐含波动率(GVZCLS)的极端跳升与衰退。黄金与美债同属终极避险资产，当 GVZCLS 短期内边际暴涨，代表地缘或通胀恐慌达到极致，此时流动性挤兑往往导致美债(TLT)惨遭现金化错杀。当此极端避险脉冲开始衰竭（期权波动率回落）时，说明非理性恐慌释放，避险资金重新稳定配置回长端美债，此时为绝佳的做多脉冲点；反之，波动率异常暴跌后企稳，代表风险偏好极速扩张，资金流出美债，构成看空信号。
    数据: gvzcls (CBOE黄金ETF隐含波动率指数)
    触发: GVZ 5日变动量的 252日 Z-Score > 2.5 (极度恐慌) 且 GVZ跌破3日均线并开始回落 (二阶衰竭)
    输出: 满足恐慌衰竭为 +1.0 (看多美债)；贪婪衰竭为 -1.0 (看空美债)；常态下为 0.0
    """

    def __init__(self, zscore_window: int = 252, shock_window: int = 5, lookback: int = 3):
        self.name = 'gold_vol_shock_exhaustion'
        self.zscore_window = zscore_window
        self.shock_window = shock_window
        self.lookback = lookback

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'gvzcls' not in data.columns:
            return signal
            
        # 填充前值，保证数据连续性
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝不直接使用期权波动率水位的绝对值，而是使用短期动量，捕捉预期的突变瞬间
        gvz_momentum = gvz.diff(self.shock_window)
        
        # 滚动分布特征计算，生成经济学含义的极值标准化指标
        roll_mean = gvz_momentum.rolling(self.zscore_window, min_periods=self.zscore_window // 2).mean()
        roll_std = gvz_momentum.rolling(self.zscore_window, min_periods=self.zscore_window // 2).std()
        
        # 防止因分母为0导致Inf爆算
        roll_std = roll_std.replace(0.0, np.nan)
        zscore = (gvz_momentum - roll_mean) / roll_std
        
        # 标记近期是否出现过极致的异动，给脉冲一个观察窗口防止滞后
        extreme_high_shock = zscore.rolling(self.lookback).max() > 2.5
        extreme_low_shock = zscore.rolling(self.lookback).min() < -2.5
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 绝对禁止在单边发散时进场，必须强制附带"反向回落/拐头"的动能衰竭条件
        # 高位衰竭：当前动能必须向下且有效跌破短期支撑
        high_exhaustion = (gvz.diff(1) < 0) & (gvz < gvz.rolling(self.lookback).mean())
        # 低位企稳：当前动能必须向上且有效突破短期压力
        low_exhaustion = (gvz.diff(1) > 0) & (gvz > gvz.rolling(self.lookback).mean())
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 常态必为 0.0，只在极端事件与衰竭共振时刻输出脉冲打点
        signal[extreme_high_shock & high_exhaustion] = 1.0
        signal[extreme_low_shock & low_exhaustion] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, shock_window={self.shock_window}, lookback={self.lookback})"