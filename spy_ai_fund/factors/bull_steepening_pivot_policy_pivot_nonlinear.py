import numpy as np
import pandas as pd

class BullSteepeningPivotFactor:
    """Bull Steepening 政策转向冲量因子 (policy_pivot/nonlinear)

    逻辑: 捕捉市场预期的瞬间转向。当短端利率(DGS2)连续5天内剧烈下行超过20基点，且收益率曲线(T10Y2Y)同步陡峭化超过10基点，意味着市场在抢跑美联储的鸽派宽松转向(Bull Steepening)，此时美股通常会获得极强的流动性看多冲量。反之，短端剧烈上行且曲线平坦化代表超预期鹰派紧缩打击，输出看空脉冲。
    数据: dgs2, t10y2y
    输出: +1.0 看多（宽松抢跑脉冲），-1.0 看空（紧缩恐慌脉冲），常态 0.0
    触发条件: 基于经济学25基点单次加/降息步长，设定20基点变动为重定价极值。脉冲型触发，仅在状态翻转首日输出，预期Trigger Rate 5%左右。
    """

    def __init__(self, short_rate_threshold=0.20, curve_steep_threshold=0.10, lookback_window=5):
        self.name = 'bull_steepening_pivot_pulse'
        self.rate_thr = short_rate_threshold
        self.curve_thr = curve_steep_threshold
        self.window = lookback_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少核心数据，直接返回常态0.0信号
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 前向填充缺失值以防止计算由于节假日缺失而断裂
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 计算边际变化冲量 (动量变化)
        dgs2_diff = dgs2.diff(self.window)
        curve_diff = t10y2y.diff(self.window)

        # ==========================================
        # 1. 鸽派流动性突变: Bull Steepening (短端骤降驱动曲线变陡)
        # ==========================================
        bull_steepening = (dgs2_diff <= -self.rate_thr) & (curve_diff >= self.curve_thr)
        
        # 脉冲铁律: 仅在预期发生改变的瞬间触发 (昨日未满足，今日满足)
        long_pulse = bull_steepening & (~bull_steepening.shift(1).fillna(False))
        
        # 二阶导确认防飞刀: 转向发生当日，短端收益率不能出现超预期反抽，必须维持下行或走平状态
        long_valid = long_pulse & (dgs2.diff(1) <= 0.0)

        # ==========================================
        # 2. 鹰派流动性收紧: Bear Flattening (短端暴拉驱动曲线变平/倒挂加深)
        # ==========================================
        bear_flattening = (dgs2_diff >= self.rate_thr) & (curve_diff <= -self.curve_thr)
        
        # 脉冲铁律: 紧缩预期打满的瞬间
        short_pulse = bear_flattening & (~bear_flattening.shift(1).fillna(False))
        
        # 二阶导确认防反弹: 当日短端收益率必须处于继续上行或走平的紧缩压迫中
        short_valid = short_pulse & (dgs2.diff(1) >= 0.0)

        # 生成目标脉冲信号
        signal = pd.Series(0.0, index=data.index)
        signal[long_valid] = 1.0
        signal[short_valid] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, rate_thr={self.rate_thr})"