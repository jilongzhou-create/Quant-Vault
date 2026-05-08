import numpy as np
import pandas as pd

class FundingRatesVolReversalFactor:
    """Funding and Rates Volatility Reversal (volatility/nonlinear)

    逻辑: 结合短期资金避险需求(Fed Funds与3个月T-Bill利差)波动率和收益率曲线(10Y-2Y)波动率。当两大核心利率波动率总和达到极值且开始同步衰竭时，代表流动性紧缩与宏观重新定价的极度恐慌见顶。此时若收益率曲线出现边际趋陡(Bull Steepening)，确认美联储宽松预期已主导市场，触发做多美债的高胜率脉冲。
    数据: dff (联邦基金利率), dtb3 (3个月国债利率), t10y2y (10年-2年期限利差)
    触发: (dff-dtb3)与(t10y2y)的波动率252日Z-Score之和 > 2.5，且两者波动率均开始下穿3日均线(二阶导数衰竭)，同时10Y-2Y利差3日边际走阔(>0)。
    输出: +1.0 表示资金面与期限结构波动率极值同步衰竭，确认避险宽松反转，做多美债(TLT)。
    """

    def __init__(self):
        self.name = 'funding_rates_vol_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格遵守铁律1: 初始化全 0.0，常态下处于休眠状态
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖数据
        required_cols = ['dff', 'dtb3', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 处理缺失值 (假期等)
        dff = data['dff'].ffill()
        dtb3 = data['dtb3'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 1. 资金面避险溢价 (Funding Safe Haven Premium)
        # 在极端恐慌时，3个月国债遭疯狂抢购，收益率会剧烈低于基准政策利率
        bill_premium = dff - dtb3
        
        # 计算流动性溢价与期限结构的边际变化波动率 (21日滚动标准差)
        bp_vol = bill_premium.diff().rolling(window=21).std()
        yc_vol = t10y2y.diff().rolling(window=21).std()
        
        # 计算252日 Z-Score 衡量绝对波动率极值
        bp_vol_mean = bp_vol.rolling(window=252).mean()
        bp_vol_std = bp_vol.rolling(window=252).std()
        bp_vol_z = (bp_vol - bp_vol_mean) / bp_vol_std.replace(0.0, np.nan)
        
        yc_vol_mean = yc_vol.rolling(window=252).mean()
        yc_vol_std = yc_vol.rolling(window=252).std()
        yc_vol_z = (yc_vol - yc_vol_mean) / yc_vol_std.replace(0.0, np.nan)
        
        # 综合 FICC 核心波动率指标 (避免单一数据的假突破)
        macro_vol_sum = bp_vol_z + yc_vol_z
        
        # 条件1 (极值确认): 双重波动率必须处于历史高位区间
        cond_extreme = macro_vol_sum > 2.5
        
        # 铁律2 (二阶导数防飞刀): 波动率极度飙升时绝不接飞刀，必须等待其下穿3日均线开始衰竭
        bp_vol_exhaustion = bp_vol < bp_vol.rolling(window=3).mean()
        yc_vol_exhaustion = yc_vol < yc_vol.rolling(window=3).mean()
        cond_exhaustion = bp_vol_exhaustion & yc_vol_exhaustion
        
        # 铁律3 (边际变化确认): 恐慌瓦解后，收益率曲线必须呈现趋陡特征(短端更快回落)，证明转向宽松逻辑
        cond_steepening = t10y2y.diff(periods=3) > 0.0
        
        # 综合触发条件
        trigger = cond_extreme & cond_exhaustion & cond_steepening
        
        # 仅在同时满足极值、衰竭、趋势确认的极少数时刻输出狙击手脉冲信号
        signal[trigger] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"