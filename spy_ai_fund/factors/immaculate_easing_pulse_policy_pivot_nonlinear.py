import numpy as np
import pandas as pd

class ImmaculateEasingPulseFactor:
    """Immaculate Easing Pulse (policy_pivot/nonlinear)

    逻辑: 捕捉"完美降息/紧缩"的宏观流动性脉冲。当实际利率(dfii10)出现非线性暴跌，
          且盈亏平衡通胀预期(t10yie)并未跟随崩溃时，意味着美联储在没有严重衰退风险(通胀预期锚定)
          的情况下释放流动性，这是美股最强的估值扩张信号(看多)。反之，实际利率飙升而通胀预期平稳，
          意味着纯粹的紧缩冲击(看空)。
    数据: dfii10 (10年期TIPS实际收益率), t10yie (10年期盈亏平衡通胀率)
    输出: +1.0 (完美宽松脉冲), -1.0 (鹰派紧缩脉冲), 0.0 (常态)
    触发条件: 实际收益率5天显著极值变动，且通胀预期未向不利方向极度偏移。
              通过一阶差分的跃迁点强制实现脉冲(Sniper Trigger)，预期Trigger Rate 6% - 10%。
    """

    def __init__(self):
        self.name = 'immaculate_easing_pulse_policy_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全0信号 (严格遵守零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖数据是否存在
        if 'dfii10' not in data.columns or 't10yie' not in data.columns:
            return signal

        # 填充缺失值(处理法定节假日)
        df_real = data['dfii10'].ffill()
        df_brk = data['t10yie'].ffill()

        # 计算5个交易日的边际动量变化 (遵守边际变化铁律)
        real_diff_5d = df_real.diff(5)
        brk_diff_5d = df_brk.diff(5)

        # 计算滚动的历史Z-Score，识别极端定价偏离
        # 使用 252 个交易日 (约一年) 的滚动窗口，最小观测期 60 天
        real_diff_mean = real_diff_5d.rolling(window=252, min_periods=60).mean()
        real_diff_std = real_diff_5d.rolling(window=252, min_periods=60).std()
        real_z = (real_diff_5d - real_diff_mean) / (real_diff_std + 1e-8)

        brk_diff_mean = brk_diff_5d.rolling(window=252, min_periods=60).mean()
        brk_diff_std = brk_diff_5d.rolling(252, min_periods=60).std()
        brk_z = (brk_diff_5d - brk_diff_mean) / (brk_diff_std + 1e-8)

        # 构建非线性交叉逻辑
        # 看多脉冲 (+1.0): 
        # 1. 实际利率剧烈下行 (Z < -1.25, 约前10%尾部概率) 且 绝对降幅超过10个基点 (过滤噪音)
        # 2. 通胀预期没有崩溃 (Z > -0.75)，排除了因严重经济衰退恐慌导致的利率下行
        bull_cond = (real_z < -1.25) & (real_diff_5d < -0.10) & (brk_z > -0.75)

        # 看空脉冲 (-1.0): 
        # 1. 实际利率剧烈上升 (Z > 1.25) 且 绝对升幅超过10个基点
        # 2. 通胀预期并未飙升 (Z < 0.75)，说明这是美联储主动超预期收紧，而不是单纯的通胀失控补偿
        bear_cond = (real_z > 1.25) & (real_diff_5d > 0.10) & (brk_z < 0.75)

        # 严格执行脉冲铁律: 仅在条件"刚刚满足"的瞬间触发 (上升沿触发)
        bull_pulse = bull_cond & (~bull_cond.shift(1).fillna(False))
        bear_pulse = bear_cond & (~bear_cond.shift(1).fillna(False))

        # 赋值信号
        signal.loc[bull_pulse] = 1.0
        signal.loc[bear_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"