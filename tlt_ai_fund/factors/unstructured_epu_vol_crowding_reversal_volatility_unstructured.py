import numpy as np
import pandas as pd

class VolatilityFomcRegimeFactor:
    """Volatility / Unstructured FOMC Regime Factor

    逻辑: VIX波动率极值衰竭与美债的关联并非静态。传统因子“VIX衰竭=做多美债”会产生毒性边缘贡献，因为多数时候VIX衰竭代表风险偏好回归(Risk-on)，导致股涨债跌。本因子创新性地引入美联储非结构化NLP情感动量(fomc_sentiment)进行宏观状态切割：
    当美联储边际转鸽(fomc_mom>0)时, 波动率极值衰竭代表"鸽派软着陆预期"(利多美债), 波动率飙升代表"避险情绪"(利多美债)。
    反之, 若美联储边际转鹰(fomc_mom<0), 波动率衰竭代表"对高利率的脱敏"(利空美债), 波动率飙升代表"股债双杀的滞胀恐慌"(利空美债)。
    此逻辑完全解耦了波动率因子内部的自相关摩擦，提供纯粹的FICC正向Alpha。
    
    数据: vixcls (市场波动率), fomc_sentiment (NLP央行情感得分)
    触发: VIX处于极值(126日Z-Score)且发生短期二阶动能反转，同时FOMC季度动量发生定向偏转
    输出: 脉冲信号, +1.0 看多美债, -1.0 看空美债
    """

    def __init__(self):
        self.name = 'vol_fomc_regime_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值信号 (严格遵守零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)

        # 校验所需数据缺失情况
        if 'vixcls' not in data.columns or 'fomc_sentiment' not in data.columns:
            return signal

        vix = data['vixcls'].ffill()
        fomc = data['fomc_sentiment'].ffill()

        # 1. 波动率水平极值 (使用126日半年度Z-Score捕捉中短期宏观极值)
        vix_mean = vix.rolling(window=126).mean()
        vix_std = vix.rolling(window=126).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)

        # 2. 波动率二阶导数 (严格遵守不接飞刀铁律：必须包含均值回归与动能变化)
        vix_ma5 = vix.rolling(window=5).mean()
        vix_diff3 = vix.diff(3)
        
        # 衰竭条件: 极值高位开始实质性回落
        vix_exhausting = (vix < vix_ma5) & (vix_diff3 < -0.5) 
        # 苏醒条件: 极值低位开始实质性飙升
        vix_waking_up = (vix > vix_ma5) & (vix_diff3 > 0.5)   

        # 3. NLP情感边际变化 (严格遵守边际变化铁律：绝对禁止使用绝对值)
        # 使用 63个交易日(约3个月) 的差分，正好平滑对比过去1到2次FOMC会议的情感基调转变
        # 变动量 > 0.05 代表边际转鸽, < -0.05 代表边际转鹰
        fomc_mom = fomc - fomc.shift(63)

        # 4. 多空脉冲逻辑生成 (条件必须同时满足以生成 Sniper Pulse)
        
        # 【多头脉冲】
        # A. 鸽派软着陆: 极度恐慌后VIX衰竭 + 联储已提前转鸽 -> 流动性宽松预期确认 (看多)
        long_cond1 = (vix_z > 1.2) & vix_exhausting & (fomc_mom > 0.05)
        # B. 鸽派避险: 市场极度自满被打破 + 联储已提前转鸽 -> 纯粹避险模式开启, 无加息担忧 (看多)
        long_cond2 = (vix_z < -0.8) & vix_waking_up & (fomc_mom > 0.05)

        # 【空头脉冲】
        # A. 鹰派抗跌: 极度恐慌后VIX衰竭 + 联储依然边际转鹰 -> 市场对高利率脱敏，无风险利率继续飙升 (看空)
        short_cond1 = (vix_z > 1.2) & vix_exhausting & (fomc_mom < -0.05)
        # B. 滞胀恐慌: 市场自满被打破 + 联储边际转鹰 -> 典型滞胀危机，股债双杀被触发(如2022) (看空)
        short_cond2 = (vix_z < -0.8) & vix_waking_up & (fomc_mom < -0.05)

        # 赋值脉冲信号
        signal.loc[long_cond1 | long_cond2] = 1.0
        signal.loc[short_cond1 | short_cond2] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"