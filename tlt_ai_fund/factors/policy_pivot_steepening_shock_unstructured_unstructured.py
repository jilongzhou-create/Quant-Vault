import numpy as np
import pandas as pd

class UnstructuredFomcDipExhaustionFactor:
    """FOMC Sentiment Dip Exhaustion Factor (unstructured/unstructured)

    逻辑: 捕捉美联储情绪与市场短期定价的背离。当 FOMC 边际转鸽时 (宏观趋势看多美债), 若市场因短期噪音出现收益率飙升 (局部鹰派脉冲), 则在收益率飙升衰竭时买入 TLT。此为顺宏观大势、逆短期情绪的高胜率脉冲策略。避免了趋势末端追高的胜率塌陷问题。
    数据: fomc_sentiment (非结构化鸽鹰得分), dgs2 (2年期美债收益率)
    触发: fomc_sentiment 边际转鸽 + dgs2 3日变动 > 1.0σ + dgs2 今日变动 < 0 (衰竭)
    输出: +1.0 (顺势抄底看多 TLT), -1.0 (顺势做空看空 TLT)
    """

    def __init__(self):
        self.name = 'unstructured_fomc_dip_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 检查是否包含所需的核心列
        required_cols = ['fomc_sentiment', 'dgs2']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        fomc = data['fomc_sentiment'].ffill()
        dgs2 = data['dgs2'].ffill()
        
        # 1. 边际变化铁律: 宏观基调 (FOMC Sentiment Marginal Trend)
        # 对阶梯状的 FOMC 情绪得分进行 21 日 EMA 平滑，并计算 10 日动量，提取明确的边际转向信号
        fomc_ema = fomc.ewm(span=21, min_periods=10).mean()
        fomc_trend = fomc_ema.diff(10)
        
        # 2. 短端利率的脉冲变化 (Local Market Pricing Shocks)
        dgs2_1d_chg = dgs2.diff(1)
        dgs2_3d_chg = dgs2.diff(3)
        
        # 利用 63 日滚动波动率衡量 3 日收益率变动的动态标准差
        dgs2_3d_vol = dgs2_1d_chg.rolling(63, min_periods=21).std() * np.sqrt(3)
        
        # 3. 捕捉背离极值: 寻找与宏观基调相反的短期脉冲 (Z-Score > 1.0)
        local_hawkish_shock = dgs2_3d_chg > (dgs2_3d_vol * 1.0)
        local_dovish_shock = dgs2_3d_chg < -(dgs2_3d_vol * 1.0)
        
        # 4. 二阶导数铁律: 防接飞刀衰竭条件 (Second Derivative Exhaustion)
        # 收益率飙升衰竭: 今日下跌，且动能弱于过去 3 日均值
        yield_rise_exhausted = (dgs2_1d_chg < 0.0) & (dgs2_1d_chg < dgs2_1d_chg.rolling(3).mean())
        # 收益率暴跌衰竭: 今日上涨，且动能强于过去 3 日均值 (负值反弹)
        yield_drop_exhausted = (dgs2_1d_chg > 0.0) & (dgs2_1d_chg > dgs2_1d_chg.rolling(3).mean())
        
        # 5. 狙击手零值休眠触发 (Signal Triggers)
        # 买入 TLT (+1.0): 联储边际转鸽，但市场因噪音出现了鹰派暴跌，在暴跌动能衰竭时精准抄底
        buy_cond = (fomc_trend > 0.01) & local_hawkish_shock & yield_rise_exhausted
        
        # 做空 TLT (-1.0): 联储边际转鹰，但市场因噪音出现了鸽派反弹，在反弹动能衰竭时精准做空
        short_cond = (fomc_trend < -0.01) & local_dovish_shock & yield_drop_exhausted
        
        signal[buy_cond] = 1.0
        signal[short_cond] = -1.0
        
        # 处理预热期的 NaN
        signal[fomc_trend.isna() | dgs2_3d_vol.isna()] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"