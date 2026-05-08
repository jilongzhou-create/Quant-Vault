import numpy as np
import pandas as pd

class BullSteepeningPulseFactor:
    """美联储政策转向与多头陡峭化脉冲因子 (policy_pivot/nonlinear)

    逻辑: 捕捉市场对美联储货币政策预期发生剧变的极短窗口。真正的买点出现在'短端利率(2年期)剧烈下行导致曲线突然变陡'(Bull Steepening)的瞬间，
          此时市场抢跑降息预期，流动性冲量瞬间爆发，利好美股。反之，加息恐慌导致的'空头平坦化'(Bear Flattening)视为恶化信号。
    数据: dgs2 (2年期美债), t10y2y (10年-2年期限利差)
    输出: +1.0 看多(抢跑降息/多头陡峭化), -1.0 看空(加息恐慌/空头平坦化), 0.0 常态观望
    触发条件: 5天内2年期收益率Z-Score极度下行 且 期限利差极度陡峭化，且当日动量未反转。预期 Trigger Rate 控制在 8% 左右。
    """

    def __init__(self, window=252, z_threshold=1.25, min_bp_change=0.05):
        # 命名规范: snake_case
        self.name = 'policy_pivot_bull_steepening_pulse'
        # 回溯窗口 252 个交易日(约1年)
        self.window = window
        # 极值判定: 1.25个标准差(约分布的前10%极值)
        self.z_threshold = z_threshold
        # 最低经济学显著阈值: 5个基点(避免在死水微澜的市场中被极小的噪音触发Z-Score极值)
        self.min_bp_change = min_bp_change

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律: 默认返回0.0, 只在极端事件当天脉冲触发
        signal = pd.Series(0.0, index=data.index)
        
        # 铁律: 必须处理缺失情况
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return signal

        # 铁律: 边际变化(禁止使用绝对水位)! 关注短端利率和利差的5日动量冲量
        dgs2_diff5 = data['dgs2'].diff(5)
        t10y2y_diff5 = data['t10y2y'].diff(5)
        
        # 动量延续/衰竭确认: 确保当天的变化与冲量方向一致，防止在反转当天接飞刀
        dgs2_diff1 = data['dgs2'].diff(1)
        
        # 滚动 Z-Score 计算历史极值，适应不同时期的波动率环境
        dgs2_roll_mean = dgs2_diff5.rolling(self.window).mean()
        dgs2_roll_std = dgs2_diff5.rolling(self.window).std()
        dgs2_z = (dgs2_diff5 - dgs2_roll_mean) / (dgs2_roll_std + 1e-8)
        
        t10y2y_roll_mean = t10y2y_diff5.rolling(self.window).mean()
        t10y2y_roll_std = t10y2y_diff5.rolling(self.window).std()
        t10y2y_z = (t10y2y_diff5 - t10y2y_roll_mean) / (t10y2y_roll_std + 1e-8)

        valid_idx = dgs2_z.notna() & t10y2y_z.notna()

        # === 脉冲触发逻辑 ===
        
        # 强看多脉冲 (+1.0): Bull Steepening (多头陡峭化)
        # 逻辑: 2年期极具下坠 (Z < -1.25) AND 利差急剧走阔 (Z > 1.25)
        bull_pulse = (
            (dgs2_z < -self.z_threshold) & 
            (t10y2y_z > self.z_threshold) & 
            (dgs2_diff5 < -self.min_bp_change) &  # 绝对值保护，至少下跌5个基点
            (t10y2y_diff5 > self.min_bp_change) &
            (dgs2_diff1 <= 0.0)                   # 动量要求：当天短端利率依然处于下行或衰竭停滞状态
        )

        # 轻微看空/趋势恶化脉冲 (-1.0): Bear Flattening (空头平坦化)
        # 逻辑: 2年期急剧飙升 (Z > 1.25) AND 利差急剧收窄/倒挂加深 (Z < -1.25)
        bear_pulse = (
            (dgs2_z > self.z_threshold) & 
            (t10y2y_z < -self.z_threshold) & 
            (dgs2_diff5 > self.min_bp_change) &   # 绝对值保护，至少飙升5个基点
            (t10y2y_diff5 < -self.min_bp_change) &
            (dgs2_diff1 >= 0.0)                   # 动量要求：当天短端利率依然在飙升或高位维持
        )

        # 赋值非零脉冲
        signal.loc[valid_idx & bull_pulse] = 1.0
        signal.loc[valid_idx & bear_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"BullSteepeningPulseFactor(window={self.window}, z_threshold={self.z_threshold})"