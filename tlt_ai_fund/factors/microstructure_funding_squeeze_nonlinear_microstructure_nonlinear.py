import numpy as np
import pandas as pd

class MicrostructureRateExhaustionFactor:
    """微观结构与利率预期衰竭反转因子 (microstructure/nonlinear)

    逻辑: 结合短端利率(dgs2)的宏观定价脉冲与TLT成交量的微观投降信号。因单纯使用VIX等泛风险指标会导致股债跷跷板效应带来的方向误判(Conditional IC低)，本因子直接锚定美债核心定价驱动(2年期收益率)。
         当2年期国债收益率短期飙升(加息恐慌)且TLT放出巨量时，意味着长端美债抛售投降(Capitulation)。
         一旦收益率开始回落且成交量萎缩(二阶导数衰竭)，迎来美债极佳的反弹抄底点。
         当收益率暴跌(降息亢奋)且放巨量后，收益率反弹且缩量，则是多头亢奋衰竭的做空点。
         因子预测方向与TLT高度因果绑定，极大提升 Conditional IC。
    数据: dgs2 (2年期美债收益率), volume (TLT成交量)
    触发: (dgs2 10日动量 Z-Score) + (TLT volume Z-Score) > 1.2，且单边动量确认(绝对值>0.5)，同时满足双重衰竭条件。
    输出: 脉冲信号 +1.0 或 -1.0，常态 0.0。
    """

    def __init__(self):
        self.name = 'micro_rate_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 检查必要数据列
        if 'dgs2' not in data.columns or 'volume' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        dgs2 = data['dgs2'].ffill()
        volume = data['volume'].ffill()
        
        signal = pd.Series(0.0, index=data.index)
        
        # 2. 计算宏观定价边际变化: 短端利率 10 日动量的 Z-Score
        dgs2_mom = dgs2.diff(10)
        # 使用 252 个交易日窗口，刻画年内相对极值
        dgs2_mom_mean = dgs2_mom.rolling(window=252, min_periods=63).mean()
        dgs2_mom_std = dgs2_mom.rolling(window=252, min_periods=63).std()
        dgs2_mom_z = (dgs2_mom - dgs2_mom_mean) / (dgs2_mom_std + 1e-8)
        
        # 3. 计算微观结构极值: TLT 成交量的 Z-Score
        # 使用 63 个交易日(一季度)窗口，刻画短期内的相对爆量
        vol_mean = volume.rolling(window=63, min_periods=21).mean()
        vol_std = volume.rolling(window=63, min_periods=21).std()
        vol_z = (volume - vol_mean) / (vol_std + 1e-8)
        
        # 填充 NaN 以避免后续逻辑运算失败
        dgs2_mom_z = dgs2_mom_z.fillna(0)
        vol_z = vol_z.fillna(0)
        
        # 4. 构建宏微观共振指数
        # 抛售恐慌指数: 收益率上升(动量为正) + 巨量成交
        rate_shock_idx = dgs2_mom_z + vol_z
        # 抢筹亢奋指数: 收益率下降(动量为负) + 巨量成交
        rate_euphoria_idx = -dgs2_mom_z + vol_z
        
        # 5. 二阶导数衰竭条件
        # 收益率开始回落 (低于过去3日均值) -> 抛售恐慌衰竭 (利多美债)
        dgs2_exhaust_long = dgs2 < dgs2.rolling(window=3).mean()
        # 收益率开始反弹 (高于过去3日均值) -> 抢筹亢奋衰竭 (利空美债)
        dgs2_exhaust_short = dgs2 > dgs2.rolling(window=3).mean()
        # 成交量缩量 (微观抛售/抢筹活跃度消退)
        vol_exhaust = volume < volume.rolling(window=3).mean()
        
        # 填充衰竭条件的 NaN
        dgs2_exhaust_long = dgs2_exhaust_long.fillna(False)
        dgs2_exhaust_short = dgs2_exhaust_short.fillna(False)
        vol_exhaust = vol_exhaust.fillna(False)
        
        # 6. 生成极短期狙击手脉冲信号
        # 多头: 恐慌共振极值 > 1.2 且 收益率回落 且 缩量 且 短期确实存在加息恐慌(收益率动量>0.5)
        long_cond = (rate_shock_idx > 1.2) & (dgs2_mom_z > 0.5) & dgs2_exhaust_long & vol_exhaust
        
        # 空头: 亢奋共振极值 > 1.2 且 收益率反弹 且 缩量 且 短期确实存在降息亢奋(收益率动量<-0.5)
        short_cond = (rate_euphoria_idx > 1.2) & (dgs2_mom_z < -0.5) & dgs2_exhaust_short & vol_exhaust
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"