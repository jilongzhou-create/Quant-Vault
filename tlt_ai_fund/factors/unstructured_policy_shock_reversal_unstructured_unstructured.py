import numpy as np
import pandas as pd

class UnstructuredFomcMacdExhaustionFactor:
    """FOMC Sentiment MACD Exhaustion (Unstructured/NLP)

    逻辑: 将低频阶梯状的FOMC情绪得分转化为连续的MACD动量，捕捉美联储政策预期的极端单向拥挤。当鹰派动量达到历史极端(Z-Score < -1.5)且动量开始衰竭回升时，标志着"鹰派见顶(Peak Hawkishness)"，此时市场往往已经过度Price-in加息预期，超卖美债，输出做多信号；反之，当鸽派动量极端且开始回落时，标志着"鸽派见顶"，输出做空信号。通过情绪的二阶导衰竭提前捕捉价格反转，提供区别于量价趋势的边际Alpha。
    数据: fomc_sentiment (FOMC鹰鸽情绪得分, 1.0=极度鸽派=看多美债, -1.0=极度鹰派)
    触发: MACD(21, 126)的252日Z-Score极值(>1.5或<-1.5)，且突破5日信号线且动量真实反转(diff(3)反转)
    输出: 脉冲型信号(多日聚类脉冲)，常态为0.0，极端衰竭期内输出+1.0或-1.0
    """

    def __init__(self):
        self.name = 'unstructured_fomc_macd_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化基准零值信号 (铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 缺失字段处理
        if 'fomc_sentiment' not in data.columns:
            return signal
            
        # 前向填充阶梯数据，初始值用0填充(中性防空)
        fomc = data['fomc_sentiment'].ffill().fillna(0.0)
        
        # 铁律3: 边际变化。使用 MACD 提取 NLP 情绪得分的边际变化动量，彻底解决低频数据的阶梯函数陷阱
        # span=21(约1个月)代表短期预期变化，span=126(约半年)代表中长期预期基准
        short_ema = fomc.ewm(span=21, adjust=False).mean()
        long_ema = fomc.ewm(span=126, adjust=False).mean()
        macd = short_ema - long_ema
        
        # 计算 252 日 Z-Score，定位极端情绪拥挤
        macd_mean = macd.rolling(window=252, min_periods=126).mean()
        macd_std = macd.rolling(window=252, min_periods=126).std()
        macd_std = macd_std.replace(0, np.nan) # 防止除零
        macd_z = (macd - macd_mean) / macd_std
        
        # 铁律2: 二阶导数(防接飞刀)。需等待单边情绪极值且动量开始衰竭
        macd_signal = macd.ewm(span=5, adjust=False).mean()
        macd_diff = macd.diff(3)
        
        # 鹰派动量见顶衰竭: 动量 Z-Score < -1.5 且 MACD 向上突破5日信号线，且真实差分为正
        peak_hawkish = ((macd_z < -1.5) & (macd > macd_signal) & (macd_diff > 0)).fillna(False)
        
        # 鸽派动量见顶衰竭: 动量 Z-Score > 1.5 且 MACD 向下突破5日信号线，且真实差分为负
        peak_dovish = ((macd_z > 1.5) & (macd < macd_signal) & (macd_diff < 0)).fillna(False)
        
        # 生成狙击手级聚类脉冲，将 Trigger Rate 控制在目标范围 5%~15% 内
        signal.loc[peak_hawkish] = 1.0
        signal.loc[peak_dovish] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"