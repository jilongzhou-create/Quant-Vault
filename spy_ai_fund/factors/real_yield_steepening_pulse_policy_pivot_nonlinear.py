import numpy as np
import pandas as pd

class RealYieldSteepeningPulseFactor:
    """实际利率陡峭化脉冲因子 (policy_pivot/nonlinear)

    逻辑: 实际利率是美股资产定价的重力。美联储超预期转鸽的最强确认是实际利率(DFII5)短期内剧烈下行，且由于短端抢跑降息导致伴随收益率曲线陡峭化(T10Y2Y走阔)。此窗口为抢跑流动性放松的极佳脉冲买点；反之为鹰派恐慌抛售点。
    数据: dfii5 (5年期TIPS实际利率), t10y2y (10年期与2年期国债利差)
    输出: +1.0 看多 (实际利率暴跌+曲线变陡)，-1.0 看空 (实际利率暴涨+曲线平坦/倒挂加深)，平时 0.0
    触发条件: 5年期实际利率5日变动<=-20bps 且 利差5日变动>=15bps 触发看多，结合当日动量延续防接飞刀。预期 Trigger Rate 约 5% 到 10%。
    """

    def __init__(self):
        self.name = 'real_yield_steepening_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据缺失保护
        if 'dfii5' not in data.columns or 't10y2y' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        # 向前填充缺失值以保持宏观数据的最新状态
        dfii5 = data['dfii5'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 计算一个交易周(5天)内的宏观预期极值冲量
        # 0.20 代表 20 个基点，实际利率短期变动20bps是显著的流动性冲击阈值
        # 0.15 代表 15 个基点，是判定收益率曲线发生实质性陡峭/平坦变化的经济学阈值
        dfii5_diff_5 = dfii5.diff(5)
        t10y2y_diff_5 = t10y2y.diff(5)
        
        # 1日边际动量，用于确保发车当天动量没有发生逆转(防逆势飞刀)
        dfii5_diff_1 = dfii5.diff(1)
        
        # 做多脉冲: 鸽派突变抢跑期 (流动性变宽 + 曲线急速陡峭化)
        long_cond = (
            (dfii5_diff_5 <= -0.20) & 
            (t10y2y_diff_5 >= 0.15) & 
            (dfii5_diff_1 < 0)
        )
        
        # 做空脉冲: 鹰派恐慌期 (流动性急速收紧 + 曲线急剧倒挂/平坦)
        short_cond = (
            (dfii5_diff_5 >= 0.20) & 
            (t10y2y_diff_5 <= -0.15) & 
            (dfii5_diff_1 > 0)
        )
        
        # 初始化为0.0，满足休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 触发极端脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"