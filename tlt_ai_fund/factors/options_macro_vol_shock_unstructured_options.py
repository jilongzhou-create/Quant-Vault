import numpy as np
import pandas as pd

class PolicyPivotShockExhaustionFactor:
    """政策预期突变衰竭因子 (unstructured/options)

    逻辑: 捕捉美联储政策预期(映射STIR期权市场的剧烈重定价)的极端跳跃。根据零值休眠和二阶导数铁律，当短端利率(DGS2)暴跌且曲线(T10Y2Y)极度变陡，且该突变动量开始衰竭时，确认降息周期确立并逢回调买入美债。鹰派突变同理做空。
    数据: dgs2 (2年期国债收益率), t10y2y (10年-2年利差)
    触发: 政策冲击合成 Z-Score > 2.5 且动量跌破3日均值 -> +1.0 (鸽派突变)
         政策冲击合成 Z-Score < -2.5 且动量突破3日均值 -> -1.0 (鹰派突变)
    输出: 脉冲型信号，在极端冲击衰竭时输出方向信号并保持3天。
    """

    def __init__(self):
        self.name = 'policy_pivot_shock_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse)，常态下必须为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 必须处理缺失字段的情况
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 铁律3: 边际变化。绝对禁止使用水位！计算5日动量，捕捉预期的"瞬时改变"
        dgs2_mom = dgs2.diff(5)
        curve_mom = t10y2y.diff(5)
        
        # 计算 252日(一年) Z-Score，捕捉宏观级别的极值冲击
        dgs2_z = (dgs2_mom - dgs2_mom.rolling(252).mean()) / dgs2_mom.rolling(252).std()
        curve_z = (curve_mom - curve_mom.rolling(252).mean()) / curve_mom.rolling(252).std()
        
        # 合成"政策转向冲击指数" (鸽派突变为正，鹰派突变为负)
        # 鸽派 = 短端急跌(dgs2_z < 0) + 曲线变陡(curve_z > 0) -> 导致 (curve_z - dgs2_z) 产生巨大正向极值
        pivot_shock_raw = curve_z - dgs2_z
        
        # 二次标准化，使其具备真正的 Z-Score 统计属性
        pivot_mean = pivot_shock_raw.rolling(252).mean()
        pivot_std = pivot_shock_raw.rolling(252).std()
        pivot_zscore = (pivot_shock_raw - pivot_mean) / pivot_std
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 计算3日均线用于判断情绪动量的衰竭反转
        pivot_ma3 = pivot_zscore.rolling(3).mean()
        
        # ----- 场景 A: 鸽派冲击 (降息预期确立) -----
        # 条件1: 极值 (Z-Score > 2.5 铁律)
        dovish_extreme = pivot_zscore > 2.5
        # 条件2: 衰竭 (冲击到达顶峰并开始回落，避免追高接飞刀)
        dovish_exhaustion = pivot_zscore < pivot_ma3
        long_trigger = dovish_extreme & dovish_exhaustion
        
        # ----- 场景 B: 鹰派冲击 (加息预期确立) -----
        # 条件1: 极值
        hawkish_extreme = pivot_zscore < -2.5
        # 条件2: 衰竭 (冲击到达底部并开始反弹，避免底部的流动性恐慌杀跌)
        hawkish_exhaustion = pivot_zscore > pivot_ma3
        short_trigger = hawkish_extreme & hawkish_exhaustion
        
        # 信号脉冲化: 极端事件发生当天及随后2天内输出信号
        # 确保 Trigger Rate 落在 5% - 15% 的黄金区间，同时维持狙击手特性
        long_pulse = long_trigger | long_trigger.shift(1).fillna(False) | long_trigger.shift(2).fillna(False)
        short_pulse = short_trigger | short_trigger.shift(1).fillna(False) | short_trigger.shift(2).fillna(False)
        
        # 冲突过滤与赋值
        long_final = long_pulse & (~short_pulse)
        short_final = short_pulse & (~long_pulse)
        
        signal[long_final] = 1.0
        signal[short_final] = -1.0
        
        # 清理由于滚动窗口产生的初始 NaN，确保合规
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"