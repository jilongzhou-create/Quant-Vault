import numpy as np
import pandas as pd

class DovishGoldilocksLiquidityPulseFactor:
    """政策转向与流动性冲量 (policy_pivot/nonlinear)

    逻辑: 当短期利率(DGS2)快速下行且收益率曲线变陡(T10Y2Y上升)，同时市场恐慌情绪回落(VIX下降)时，标志着市场预期'软着陆+预防性降息'的金发女孩环境，强烈看多；反之，若短端利率飙升引发VIX上涨，则为鹰派紧缩恐慌，看空。
    数据: dgs2, t10y2y, vixcls
    输出: 脉冲信号 [-1.0, 1.0]
    触发条件: 5日内短端利率波动超5个基点(0.05%)且收益率曲线配合变化，辅以VIX同向验证。预期Trigger Rate 8%-12%。
    """

    def __init__(self):
        self.name = 'dovish_goldilocks_liquidity_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 核心字段检查
        required_cols = ['dgs2', 't10y2y', 'vixcls']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index)
        
        # 填充缺失值并提取所需列
        df = data[required_cols].ffill()
        
        # 计算5个交易日(约一周)的边际变化量
        # 经济学含义: 5天内DGS2变化0.05(5bp)，代表市场对单次加息/降息(25bp)预期发生了约20%的显著概率重估
        dgs2_diff = df['dgs2'].diff(5)
        t10y2y_diff = df['t10y2y'].diff(5)
        vix_diff = df['vixcls'].diff(5)
        
        signal = pd.Series(0.0, index=data.index)
        
        # Bullish (+1.0): 鸽派转向预期 + 曲线牛陡 + 恐慌衰退 -> 资金重返风险资产 (非衰退式降息预期)
        bull_steepening = (dgs2_diff <= -0.05) & (t10y2y_diff >= 0.01)
        risk_on = vix_diff < 0.0
        bullish = bull_steepening & risk_on
        
        # Bearish (-1.0): 鹰派惊吓 + 曲线熊平 + 恐慌升温 -> 资金撤离风险资产 (紧缩恐慌)
        bear_flattening = (dgs2_diff >= 0.05) & (t10y2y_diff <= -0.01)
        risk_off = vix_diff > 0.0
        bearish = bear_flattening & risk_off
        
        # 生成脉冲信号
        signal[bullish] = 1.0
        signal[bearish] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"