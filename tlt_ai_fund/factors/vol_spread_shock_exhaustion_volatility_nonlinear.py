import numpy as np
import pandas as pd

class YieldCurveVelocityExhaustionFactor:
    """Yield Curve Velocity Exhaustion (volatility/nonlinear)

    逻辑: 捕捉收益率曲线(t10y2y)的极端陡峭化或平坦化脉冲(恐慌性降息/加息定价)。当这种极端定价的动能开始衰竭时，代表短期拥挤交易解体，触发TLT的反转脉冲。完全规避跨资产权益波动率(VIX)以确保正交性，聚焦美债市场内生的期限结构动能。
    数据: t10y2y (期限利差), dgs2 (2年期国债收益率)
    触发: 期限利差5日动能的Z-Score > 2.5 或 < -2.5 (极值)，且短端利率方向确认，随后曲线变化率反转且突破/跌破3日均线(衰竭)。
    输出: 脉冲型信号。加息恐慌导致平坦化且衰竭时做多TLT(+1.0)，降息恐慌导致陡峭化且衰竭时做空TLT(-1.0)。常态为0.0。
    """

    def __init__(self):
        self.name = 'yield_curve_vel_exhaustion_volatility_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 校验所需字段是否存在，若无则返回全0
        if 't10y2y' not in data.columns or 'dgs2' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 前向填充缺失值
        spread = data['t10y2y'].ffill()
        dgs2 = data['dgs2'].ffill()

        # 铁律3: 边际变化 (Marginal Change) -> 捕捉瞬间动能而非绝对水位
        spread_vel = spread.diff(5)
        dgs2_vel = dgs2.diff(5)

        # 计算动能的 Z-Score (使用63个交易日即一个季度的滚动窗口，确保捕捉局部的极端恐慌脉冲)
        roll_mean = spread_vel.rolling(63).mean()
        roll_std = spread_vel.rolling(63).std()
        
        # 避免除以0
        z_spread_vel = (spread_vel - roll_mean) / roll_std.replace(0, np.nan)

        # 铁律1 & 2: 二阶导数 (极值 + 衰竭) 绝对禁止接飞刀
        
        # --- 场景A: 剧烈加息恐慌 (Bear Flattener) ---
        # 条件1 (极值): 曲线剧烈平坦化 (Z < -2.5) 且 短端利率飙升确认加息恐慌 (dgs2_vel > 0)
        bear_flat_extreme = (z_spread_vel < -2.5) & (dgs2_vel > 0.0)
        # 条件2 (衰竭): 曲线停止平坦化并开始反弹，且突破3日均线
        bear_flat_exhaust = (spread.diff(1) > 0) & (spread > spread.rolling(3).mean())

        # --- 场景B: 剧烈降息恐慌 (Bull Steepener) ---
        # 条件1 (极值): 曲线剧烈陡峭化 (Z > 2.5) 且 短端利率暴跌确认降息恐慌 (dgs2_vel < 0)
        bull_steep_extreme = (z_spread_vel > 2.5) & (dgs2_vel < 0.0)
        # 条件2 (衰竭): 曲线停止陡峭化并开始回落，且跌破3日均线
        bull_steep_exhaust = (spread.diff(1) < 0) & (spread < spread.rolling(3).mean())

        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 信号赋值: 仅在极值与衰竭条件同时满足的瞬间触发脉冲
        # 加息恐慌衰竭 -> 拥挤的极右侧空头平仓 -> 美债(TLT)迎来报复性反弹 (看多 +1.0)
        signal[bear_flat_extreme & bear_flat_exhaust] = 1.0
        
        # 降息恐慌衰竭 -> 拥挤的极左侧多头平仓 -> 美债(TLT)高位回撤 (看空 -1.0)
        signal[bull_steep_extreme & bull_steep_exhaust] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"