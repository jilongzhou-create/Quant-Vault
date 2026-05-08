import numpy as np
import pandas as pd

class BullSteepeningCreditPulseFactor:
    """Bull Steepening & Credit Pulse Factor (policy_pivot/nonlinear)

    逻辑: 捕捉债券市场对美联储转向鸽派的剧烈定价(2年期短端利率急降导致收益率曲线急剧变陡)。同时通过高收益债信用利差(HY Spread)未走阔来过滤掉经济衰退恐慌(Hard Landing)。这是典型的非线性交叉：短端利率暴跌 + 信用平稳 = 预防性降息/软着陆(强看多美股)。相反则为鹰派紧缩冲击(看空)。
    数据: dgs2 (2年期国债收益率), t10y2y (10年期与2年期期限利差), bamlh0a0hym2 (ICE BofA美国高收益债利差)
    输出: +1.0 表示市场预期预防性降息的狂欢(看多)，-1.0 表示极端的鹰派紧缩预期(看空)，其余时间为 0.0
    触发条件: 2年期美债5日内急跌超15个基点 AND 曲线变陡超10个基点 AND 信用利差未显著走阔。预期 Trigger Rate 在 5% 到 10% 之间。
    """

    def __init__(self):
        self.name = 'bull_steepening_credit_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查是否包含所需字段
        required_cols = ['dgs2', 't10y2y', 'bamlh0a0hym2']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        # 向前填充处理缺失值(周末/节假日)
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()

        # 计算 5 个交易日的边际动量变化 (单位: %)
        dgs2_5d_diff = dgs2.diff(5)
        t10y2y_5d_diff = t10y2y.diff(5)
        hy_5d_diff = hy_spread.diff(5)

        signal = pd.Series(0.0, index=data.index)

        # 看多脉冲 (Bullish Pulse): 鸽派转向 + 软着陆预期
        # 1. 2年期美债收益率在5天内快速下跌超 15 bps (市场疯狂抢跑降息)
        # 2. 收益率曲线急剧变陡超 10 bps (典型的 Bull Steepening 特征)
        # 3. 高收益债信用利差边际变化 <= 5 bps (信用市场平稳，证明是保险式降息而非衰退恐慌)
        bull_cond = (
            (dgs2_5d_diff < -0.15) &
            (t10y2y_5d_diff > 0.10) &
            (hy_5d_diff <= 0.05)
        )

        # 看空脉冲 (Bearish Pulse): 鹰派冲击 + 紧缩预期
        # 1. 2年期美债收益率在5天内急升超 15 bps (超预期紧缩)
        # 2. 收益率曲线急剧变平或倒挂加深超 10 bps (Bear Flattening)
        # 3. 高收益债信用利差走阔 >= 5 bps (紧缩导致企业融资环境恶化)
        bear_cond = (
            (dgs2_5d_diff > 0.15) &
            (t10y2y_5d_diff < -0.10) &
            (hy_5d_diff >= 0.05)
        )

        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"