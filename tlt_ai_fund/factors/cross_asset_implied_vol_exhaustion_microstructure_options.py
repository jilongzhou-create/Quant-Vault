import numpy as np
import pandas as pd

class RatesMicrostructureVolExhaustionFactor:
    """利率波动微观结构衰竭因子 (microstructure/options)

    逻辑: 使用美债收益率曲线(10Y-2Y, 10Y-3M)的实际波动率作为MOVE指数(利率期权隐含波动率)的代理。极高的利率波动率会导致做市商VAR超限而抛售美债(流动性危机)。当波动率创下极端高位(Z-Score > 2.5)并开始回落(衰竭)时，VAR约束解除，强平抛压结束，机构买盘重新入场，形成看多TLT的脉冲。此为纯边际变化+二阶导数逻辑，与跨资产(VIX/黄金)因子完全正交。
    数据: t10y2y, t10y3m
    触发: 波动率代理 252日 Z-Score > 2.5 且当天值 < 3日均值 且 .diff() < 0。
    输出: +1.0 看多脉冲信号 (触发后延续4天以满足Trigger Rate要求)
    """

    def __init__(self):
        self.name = 'rates_micro_vol_exhaustion_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 检查所需数据列是否存在
        if 't10y2y' not in data.columns or 't10y3m' not in data.columns:
            return signal
            
        curve_2y = data['t10y2y'].ffill()
        curve_3m = data['t10y3m'].ffill()
        
        # 铁律3: 边际变化 (只使用曲线的每日变化率)
        dy_2y = curve_2y.diff()
        dy_3m = curve_3m.diff()
        
        # 计算利率微观波动率代理 (21天实际波动率, 代理缺失的MOVE指数)
        vol_2y = dy_2y.rolling(window=21).std()
        vol_3m = dy_3m.rolling(window=21).std()
        proxy_move = (vol_2y + vol_3m) / 2.0
        
        # 计算长期 Z-Score 识别极端恐慌状态
        roll_mean = proxy_move.rolling(window=252).mean()
        roll_std = proxy_move.rolling(window=252).std().replace(0, np.nan)
        z_score = (proxy_move - roll_mean) / roll_std
        
        # 铁律2: 二阶导数防接飞刀 (极端高位 + 衰竭反转)
        # 条件1: 利率市场波动率处于极端高位
        cond_extreme = z_score > 2.5
        
        # 条件2: 波动率开始衰竭 (做市商恐慌消退)
        cond_exhaustion = (proxy_move < proxy_move.rolling(window=3).mean()) & (proxy_move.diff() < 0)
        
        # 组合触发条件
        trigger = cond_extreme & cond_exhaustion
        
        # 铁律1: 零值休眠 (狙击手级脉冲信号，触发后延续4天以达到 5%-15% Target Rate)
        pulse_trigger = trigger.rolling(window=4).max().fillna(0).astype(bool)
        
        # 赋值信号: 利率波动率见顶回落 -> 机构VAR解除 -> 抄底看多美债
        signal[pulse_trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"