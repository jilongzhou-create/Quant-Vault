import numpy as np
import pandas as pd

class UnstructuredFomcDriftPulseFactor:
    """Unstructured FOMC Sentiment Secondary Drift Pulse (unstructured/unstructured)

    逻辑: 捕捉 FOMC 鹰鸽态度突变后的"次级趋势"(Secondary Drift)。真实的美联储政策反转需要数周才能被长端债券完全 Price-in。
          使用 3日/10日 MACD 提取 FOMC 情绪得分的边际变化(遵守边际变化铁律，将低频阶梯数据转化为高频动量)。
          当 MACD 达到极值(Z-Score > 1.2)且开始回落(遵守二阶导数/衰竭铁律)时，说明初期情绪宣泄已见顶，确认进入平稳的次级定价脉冲。
          常态下 MACD 平缓，信号严格为 0 (遵守零值休眠铁律)。
          配合前瞻性最强的短端利率(dgs2)动量确认真实的资金流向，并利用经济政策不确定性(EPU)剔除极端恐慌期的反常避险干扰。
    数据: fomc_sentiment, dgs2, usepuindxd
    触发: 
      - 看多: fomc_macd_z > 1.2 (极度鸽派突变) + macd.diff() < 0 (动量见顶衰竭) + dgs2.diff(5) < -0.01 (短端资金下行确认) + epu_z > -1.0 (排除极度自满)
      - 看空: fomc_macd_z < -1.2 (极度鹰派突变) + macd.diff() > 0 (动量见底衰竭) + dgs2.diff(5) > 0.01 (短端资金上行确认) + epu_z < 2.0 (排除无差别恐慌导致的异常避险)
    输出: [-1.0, 1.0] 的脉冲信号。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_drift_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格遵守铁律1：初始化全 0 信号 (常态休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 必须的数据列检查，缺失则返回全0休眠信号
        required_cols = ['fomc_sentiment', 'dgs2', 'usepuindxd']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 1. 边际变化铁律：计算 FOMC 情绪 MACD (提取阶梯数据的脉冲动量)
        fomc = data['fomc_sentiment'].ffill().fillna(0.0)
        ema3 = fomc.ewm(span=3, adjust=False).mean()
        ema10 = fomc.ewm(span=10, adjust=False).mean()
        macd = ema3 - ema10
        
        # 动态 Z-Score (126日/半年窗口，适应近期波幅)
        macd_roll_mean = macd.rolling(window=126, min_periods=21).mean()
        macd_roll_std = macd.rolling(window=126, min_periods=21).std()
        macd_z = (macd - macd_roll_mean) / (macd_roll_std + 1e-6)
        
        # 2. 衰竭铁律：计算二阶导数 (Anti-Catch-Falling-Knife)
        macd_diff = macd.diff(1)
        macd_decay_bull = macd_diff < 0  # 鸽派突变动量(正值)已见顶，开始回落
        macd_decay_bear = macd_diff > 0  # 鹰派突变动量(负值)已见底，开始回升
        
        # 3. 市场资金确认：利用政策高敏感度指标 DGS2 验证实际资金行为
        dgs2 = data['dgs2'].ffill().fillna(0.0)
        dgs2_mom = dgs2.diff(5)  # 5日滚动边际变化
        
        # 4. EPU 宏观不确定性环境过滤
        epu = data['usepuindxd'].ffill().fillna(0.0)
        epu_roll_mean = epu.rolling(window=126, min_periods=21).mean()
        epu_roll_std = epu.rolling(window=126, min_periods=21).std()
        epu_z = (epu - epu_roll_mean) / (epu_roll_std + 1e-6)
        
        # 组合多头脉冲条件 (看多美债 TLT)
        bull_condition = (
            (macd_z > 1.2) &              # 条件1: 处于鸽派突变极值区
            (macd_decay_bull) &           # 条件2: 动量衰竭 (反飞刀)
            (dgs2_mom < -0.01) &          # 条件3: 市场短端利率确认有效下行(降息定价)
            (epu_z > -1.0)                # 条件4: 宏观存在一定的避险需求基底
        )
        
        # 组合空头脉冲条件 (看空美债 TLT)
        bear_condition = (
            (macd_z < -1.2) &             # 条件1: 处于鹰派突变极值区
            (macd_decay_bear) &           # 条件2: 动量衰竭 (反飞刀)
            (dgs2_mom > 0.01) &           # 条件3: 市场短端利率确认实际上行(加息定价)
            (epu_z < 2.0)                 # 条件4: 排除极度恐慌危机引发的无差别买美债避险行为
        )
        
        # 触发脉冲赋值
        signal.loc[bull_condition] = 1.0
        signal.loc[bear_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"