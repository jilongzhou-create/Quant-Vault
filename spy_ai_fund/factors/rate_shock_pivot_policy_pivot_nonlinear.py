import numpy as np
import pandas as pd

class GoldilocksPivotDivergencePulseFactor:
    """Goldilocks Pivot Divergence Pulse (policy_pivot/nonlinear)

    逻辑: 捕捉名义利率与通胀预期的非线性背离。当名义利率急跌但通胀预期坚挺时, 代表纯粹的鸽派流动性释放(非衰退), 强烈看多。当名义利率飙升但通胀预期疲软时, 代表纯粹的鹰派紧缩(非经济过热), 看空。
    数据: dgs5 (5年期国债名义收益率), t5yie (5年期盈亏平衡通胀率)
    输出: 脉冲信号, +1.0 表示鸽派流动性释放(抄底), -1.0 表示鹰派紧缩恶化(看空), 常态返回 0.0
    触发条件: 5日变化的Z-Score出现特定组合(名义利率极值+通胀预期背离), 预期 Trigger Rate 约 8% - 12%
    """

    def __init__(self):
        self.name = 'goldilocks_pivot_divergence_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 信号
        signal = pd.Series(0.0, index=data.index)

        # 检查必需字段是否存在
        if 'dgs5' not in data.columns or 't5yie' not in data.columns:
            return signal

        # 提取数据并处理缺失值 (前向填充以防节假日缺失)
        df = pd.DataFrame(index=data.index)
        df['dgs5'] = data['dgs5'].ffill()
        df['t5yie'] = data['t5yie'].ffill()

        # 计算 5 日边际变化 (捕捉利率预期的阶跃动量)
        dgs5_diff = df['dgs5'].diff(5)
        t5yie_diff = df['t5yie'].diff(5)

        # 计算 126 日 (半年) 滚动 Z-Score, 以适应不同宏观波动率的基准状态
        # 避免除以 0 的情况
        roll_std_dgs5 = dgs5_diff.rolling(126).std().replace(0, np.nan)
        z_dgs5 = (dgs5_diff - dgs5_diff.rolling(126).mean()) / roll_std_dgs5

        roll_std_t5yie = t5yie_diff.rolling(126).std().replace(0, np.nan)
        z_t5yie = (t5yie_diff - t5yie_diff.rolling(126).mean()) / roll_std_t5yie

        # 核心触发逻辑
        # 看多 (Bullish Pulse):
        # 名义利率急跌 (Z < -1.2), 表明市场抢跑美联储降息预期
        # 且 通胀预期未随之大幅崩盘 (Z > -0.6), 排除通缩/衰退恐慌导致的利率下行
        # 此时为纯粹的"金发姑娘"式鸽派流动性注入
        bull_cond = (z_dgs5 < -1.2) & (z_t5yie > -0.6)
        
        # 看空 (Bearish Pulse):
        # 名义利率急升 (Z > 1.2), 表明美联储超预期鹰派
        # 且 通胀预期未随之大涨 (Z < 0.6), 排除经济过热带来的良性利率上行
        # 此时为纯粹的鹰派紧缩冲击, 实际利率飙升
        bear_cond = (z_dgs5 > 1.2) & (z_t5yie < 0.6)

        # 赋值脉冲信号
        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"