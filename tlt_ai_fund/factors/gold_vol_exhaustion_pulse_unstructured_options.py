import numpy as np
import pandas as pd

class GoldVolExhaustionPulseFactor:
    """Gold Volatility Exhaustion Pulse (Volatility Microstructure / Options)

    逻辑: 黄金隐含波动率(GVZ)是衡量避险资产期权微观恐慌溢价的极佳代理变量。当系统性危机爆发时, 黄金波动率会发生极端跳升(跨资产抛售/流动性危机); 只有当该波动率脉冲见顶并开始衰竭回落时(二阶导数翻负), 才意味着流动性冲击结束, 此时真正的宏观基本面避险资金将确定性地加速涌入美债(TLT)。此时做多可完美避开极值当天的"流动性无差别抛售"飞刀。
    数据: gvzcls (CBOE黄金ETF隐含波动率指数)
    触发: GVZ 63日(一季度) Z-Score > 2.5 (极值) 且 GVZ < 3日滚动均值 (衰竭信号)
    输出: 触发当天及随后极短2天内输出 +1.0 (看多美债脉冲), 其余常态时间严格为 0.0
    """

    def __init__(self):
        self.name = 'gold_vol_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠, 初始信号严格设为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 数据缺失保护
        if 'gvzcls' not in data.columns:
            return signal
            
        # 获取基础数据并前向填充处理少量缺失值
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 - 不看绝对水位, 计算 63个交易日(约一季度)的滚动 Z-Score 捕捉局部突变
        lookback = 63
        roll_mean = gvz.rolling(window=lookback, min_periods=30).mean()
        roll_std = gvz.rolling(window=lookback, min_periods=30).std()
        
        # 防御零标准差导致的除以零错误
        roll_std = roll_std.replace(0.0, np.nan)
        z_score = (gvz - roll_mean) / roll_std
        
        # 铁律2: 二阶导数 - 绝对禁止直接接飞刀, 必须等待波动率开始回落(低于3日均值)
        exhaustion_condition = gvz < gvz.rolling(window=3).mean()
        
        # 脉冲触发: 极值 + 衰竭 同时满足
        buy_trigger = (z_score > 2.5) & exhaustion_condition
        
        # 赋予脉冲信号
        signal.loc[buy_trigger] = 1.0
        
        # 将极端脉冲信号向后延续2天 (满足随后极短几天内维持非零值的要求, 保障合理 Trigger Rate)
        signal = signal.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"