import numpy as np
import pandas as pd

class FxRatesLiquidityPulseFactor:
    """FX Rates Liquidity Pulse (policy_pivot/nonlinear)

    逻辑: 区分美联储真正的鸽派转向与恐慌性避险。当2年期美债收益率与美元指数同时发生显著的向下脉冲时，意味着真正的货币宽松与全球流动性释放(看多)；若两者同时向上激增，则代表紧缩冲击(看空)。恐慌期间收益率暴跌但美元会因避险暴涨，本因子通过要求两者同向大幅变动，利用跨资产(汇率+利率)交叉正交验证，精准过滤出"金发姑娘"式鸽派转向，避免被避险买盘(接飞刀)误导。
    数据: dgs2 (2年期美债), dtwexbgs (美元广义指数) 或 dexuseu (欧美汇率)
    输出: +1.0 表示确认鸽派宽松释放流动性看多，-1.0 表示鹰派紧缩冲击看空
    触发条件: 5日变化率的60日滚动Z-Score均处于同向极端(Z < -1.2 或 Z > 1.2)，并结合3天休眠冷却机制，预期 Trigger Rate 5%-10%
    """

    def __init__(self):
        self.name = 'fx_rates_liquidity_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'dgs2' not in data.columns:
            return signal
            
        dgs2 = data['dgs2'].ffill()
        
        # 优先使用宽基美元指数，若无则用欧元兑美元反转替代
        if 'dtwexbgs' in data.columns and not data['dtwexbgs'].isna().all():
            usd_index = data['dtwexbgs'].ffill()
            usd_pct = usd_index.pct_change(5)
        elif 'dexuseu' in data.columns and not data['dexuseu'].isna().all():
            # dexuseu 为 USD per EUR. 升高代表欧元强、美元弱
            # 统一口径：将 usd_pct 取反，使其为正表示美元走强
            usd_index = data['dexuseu'].ffill()
            usd_pct = -usd_index.pct_change(5)
        else:
            return signal
            
        dgs2_diff = dgs2.diff(5)
        
        # 计算 60 日滚动 Z-Score，替换0以避免除以0出现无穷大
        dgs2_std = dgs2_diff.rolling(60).std().replace(0, np.nan)
        usd_std = usd_pct.rolling(60).std().replace(0, np.nan)
        
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(60).mean()) / dgs2_std
        usd_z = (usd_pct - usd_pct.rolling(60).mean()) / usd_std
        
        # 今日变化，确保变化方向在当天依然延续且有实质动量
        dgs2_daily = dgs2.diff(1)
        
        # 真正鸽派宽松：收益率大跌 且 美元大跌 (完全排除了危机避险导致的收益率跌但美元暴涨)
        bull_cond = (
            (dgs2_z < -1.2) & 
            (usd_z < -1.2) & 
            (dgs2_daily < -0.01) & 
            (usd_pct < -0.003)
        )
        
        # 真正鹰派冲击：收益率暴涨 且 美元暴涨
        bear_cond = (
            (dgs2_z > 1.2) & 
            (usd_z > 1.2) & 
            (dgs2_daily > 0.01) & 
            (usd_pct > 0.003)
        )
        
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal.loc[bull_cond] = 1.0
        raw_signal.loc[bear_cond] = -1.0
        
        # 零值休眠铁律：加入 3 天的冷却期，确保信号是离散的脉冲(Pulse)型
        active = raw_signal != 0
        is_first_pulse = active & (~active.shift(1).fillna(False)) & (~active.shift(2).fillna(False))
        
        signal = raw_signal.where(is_first_pulse, 0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"