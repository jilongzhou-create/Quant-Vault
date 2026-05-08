import numpy as np
import pandas as pd

class GoldilocksPivotPulseFactor:
    """Goldilocks Pivot Pulse (policy_pivot/nonlinear)

    逻辑: 捕捉"软着陆式"美联储政策转向预期。当市场激进定价降息(2年期美债收益率暴跌)、且通胀预期降温(给予美联储降息掩护)的同时，信用利差收窄(排除硬着陆衰退恐慌)，此时产生强烈看多脉冲。反之，滞胀式加息预期产生看空脉冲。
    数据: dgs2 (2年期美债), t5yie (5年期盈亏平衡通胀), bamlh0a0hym2 (高收益债信用利差)
    输出: +1.0 (软着陆转向，强烈看多), -1.0 (滞胀加息冲击，轻微恐慌看空)
    触发条件: 5日内2年期收益率剧烈变动(>15个基点)且通胀预期与信用利差进行同向交叉确认，要求只在预期突变的瞬间触发，预期 Trigger Rate 5%-12%
    """

    def __init__(self):
        self.name = 'goldilocks_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 检查所需字段
        required_cols = ['dgs2', 't5yie', 'bamlh0a0hym2']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 填充前值以处理节假日造成的数据缺失不对齐
        df = data[required_cols].ffill()

        # 计算5日边际动量变化 (捕捉流动性与政策预期的短波阶跃)
        # 经济学含义: 15个基点(0.15%)的周度变化是短端利率发生实质性重定价的阈值
        dgs2_chg5 = df['dgs2'].diff(5)
        t5yie_chg5 = df['t5yie'].diff(5)
        spread_chg5 = df['bamlh0a0hym2'].diff(5)

        # 当日动量(狙击手扣动扳机条件: 事件发生当日必须顺势)
        dgs2_mom = df['dgs2'].diff(1)

        # --- 触发逻辑 1: 金发女孩/软着陆转向脉冲 (强烈看多) ---
        # 1. 2年期收益率5日内暴跌超15个bp (市场抢跑鸽派降息)
        # 2. 5年期通胀预期下降或平稳 (通胀降温，美联储有降息依据)
        # 3. 高收益债利差收窄或平稳 (最关键的非线性交叉: 证明降息不是因为衰退恐慌导致的避险)
        # 4. 当日利率继续下行 (确认动量)
        long_cond = (
            (dgs2_chg5 <= -0.15) & 
            (t5yie_chg5 <= 0.0) & 
            (spread_chg5 <= 0.0) & 
            (dgs2_mom < 0)
        )

        # --- 触发逻辑 2: 滞胀式鹰派冲击脉冲 (趋势恶化看空) ---
        # 1. 2年期收益率5日内飙升超15个bp (市场定价加息或长期高息)
        # 2. 通胀预期抬升 (通胀反弹，逼迫美联储强硬)
        # 3. 高收益债利差走阔 (高息压垮信用市场，Risk-Off)
        # 4. 当日利率继续上行 (确认动量)
        short_cond = (
            (dgs2_chg5 >= 0.15) & 
            (t5yie_chg5 > 0.0) & 
            (spread_chg5 > 0.0) & 
            (dgs2_mom > 0)
        )

        # 写入脉冲信号 (默认休眠0.0)
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"