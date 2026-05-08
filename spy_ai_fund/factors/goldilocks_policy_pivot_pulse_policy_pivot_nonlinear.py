import numpy as np
import pandas as pd

class GoldilocksPolicyPivotPulseFactor:
    """金发姑娘政策转向脉冲因子 (policy_pivot/nonlinear)

    逻辑: 捕捉美联储预期剧变时刻。短端利率(DGS2)大幅急跌且通胀预期(T10YIE)未崩盘时,为"金发姑娘"降息预期(看多); 短端利率飙升且通胀预期未跟随上行时,为真实利率毒性冲击(看空)。必须等待极值动量衰竭当天触发脉冲。
    数据: dgs2(2年期美债收益率), t10yie(10年期盈亏平衡通胀率)
    输出: [-1.0, 1.0] 的脉冲信号。正值看多美股，负值看空美股。
    触发条件: DGS2的5日变化Z-score > 1.25 或 < -1.25，配合通胀预期过滤，及日度动量反转(衰竭)。预期Trigger Rate约6%-10%。
    """

    def __init__(self):
        self.name = 'goldilocks_policy_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少核心数据，直接返回全0
        if 'dgs2' not in data.columns or 't10yie' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 提取并处理缺失值 (前向填充以处理节假日或数据发布延迟)
        dgs2 = data['dgs2'].ffill()
        t10yie = data['t10yie'].ffill()

        # 计算边际动量变化 (绝对禁止使用绝对水位,严格遵循边际变化铁律)
        dgs2_5d_change = dgs2.diff(5)
        t10yie_5d_change = t10yie.diff(5)
        
        # 计算日度变化，用于捕捉"动量衰竭" (防接飞刀铁律)
        dgs2_1d_change = dgs2.diff(1)

        # 计算 Z-Score，用于动态识别"极值"状态 (避免无意义的魔法数字绝对值)
        # 使用 252 个交易日(约1年)的滚动窗口，最小周期为一个季度(60天)
        roll_mean = dgs2_5d_change.rolling(window=252, min_periods=60).mean()
        roll_std = dgs2_5d_change.rolling(window=252, min_periods=60).std()
        
        # 避免除以0
        dgs2_5d_z = (dgs2_5d_change - roll_mean) / (roll_std + 1e-6)

        signal = pd.Series(0.0, index=data.index)

        # 看多脉冲 (抄底买入): 完美鸽派转向 (软着陆预期)
        # 1. 极值状态: 2年期收益率异常暴跌 (Z < -1.25, 约10%分位数), 市场激进抢跑降息
        # 2. 交叉过滤: 通胀预期没有大幅崩盘 (变化 > -0.05%, 即下降不超过5个基点), 排除硬着陆/通缩恐慌的可能
        # 3. 动量衰竭: 今日 DGS2 停止下跌 (1日diff > 0), 债市恐慌性买入衰竭, 意味着流动性冲击完成计价，准备轮动回股市
        bull_cond = (
            (dgs2_5d_z < -1.25) & 
            (t10yie_5d_change > -0.05) & 
            (dgs2_1d_change > 0)
        )

        # 看空脉冲 (趋势恶化): 毒性鹰派冲击 (真实利率飙升)
        # 1. 极值状态: 2年期收益率异常暴涨 (Z > 1.25), 市场交易"更高更久"的紧缩政策
        # 2. 交叉过滤: 通胀预期没有跟随大幅上升 (变化 < 0.05%), 意味着名义利率的上升完全转化为真实利率的飙升(对估值极其剧毒)
        # 3. 动量衰竭: 今日 DGS2 停止上涨 (1日diff < 0), 冲击动量衰竭, 股市开始彻底计价毒性环境，开启主跌浪
        bear_cond = (
            (dgs2_5d_z > 1.25) & 
            (t10yie_5d_change < 0.05) & 
            (dgs2_1d_change < 0)
        )

        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0

        # 确保输出名字正确
        signal.name = self.name
        
        # 零值休眠铁律: 填充初始化期间的 NaN 为 0.0
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"