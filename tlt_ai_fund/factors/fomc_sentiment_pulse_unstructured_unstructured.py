import numpy as np
import pandas as pd

class UnstructuredMacroPivotPulseFactor:
    """Unstructured Macro Pivot Pulse Factor (unstructured/unstructured)

    逻辑: 捕捉美联储货币政策与宏观不确定性突变下的脉冲趋势。通过NLP情感得分(fomc_sentiment)与经济政策不确定性(EPU)的边际跳跃，结合2年期美债收益率的极端Price-in，识别政策拐点。为了避免在情绪最高点接飞刀，必须等待利率动量(二阶导数)出现衰竭(如跌势放缓反抽)时再触发信号，以极佳盈亏比顺势跟进。
    数据: fomc_sentiment, usepuindxd, dgs2, t10y2y
    触发: NLP/EPU的5日边际变化Z-Score > 2.0 或 前端利率骤降(Z-Score < -1.5)且曲线陡峭，且过去3日内发生该极值，并在今日利率变化率高于5日均值(衰竭反抽)时触发脉冲。
    输出: 仅在拐点确认并初次衰竭的瞬间输出 +1.0 (鸽派看多) 或 -1.0 (鹰派看空)，其余时间严格为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_macro_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化常态休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 必需的底层验证数据
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 1. 政策情绪 NLP 突变 (边际变化铁律)
        fomc_z = pd.Series(0.0, index=data.index)
        if 'fomc_sentiment' in data.columns:
            fomc = data['fomc_sentiment'].ffill()
            fomc_chg = fomc.diff(5)
            # 252日(约1年)滚动标准化，防除0
            fomc_std = fomc_chg.rolling(252).std().clip(lower=1e-3)
            fomc_z = (fomc_chg - fomc_chg.rolling(252).mean()) / fomc_std
            
        # 2. 经济政策不确定性突变 (边际变化铁律)
        epu_z = pd.Series(0.0, index=data.index)
        if 'usepuindxd' in data.columns:
            epu = data['usepuindxd'].ffill()
            epu_chg = epu.diff(5)
            # 63日(约1季度)滚动标准化
            epu_std = epu_chg.rolling(63).std().clip(lower=1e-3)
            epu_z = (epu_chg - epu_chg.rolling(63).mean()) / epu_std
            
        # 3. 市场真实定价的极端冲击 (Yield 动量)
        dgs2_vel = dgs2.diff(1)
        dgs2_std = dgs2_vel.rolling(63).std().clip(lower=1e-5)
        z_dgs2 = (dgs2_vel - dgs2_vel.rolling(63).mean()) / dgs2_std
        
        t10y2y_chg = t10y2y.diff(5)
        
        # --- 极端事件定义 (Extreme) ---
        # 鸽派突变：文本极度变鸽 / 政策不确定性暴增 / 短端利率暴跌且曲线变陡
        dovish_nlp = fomc_z > 2.0
        high_epu = epu_z > 2.0
        dovish_market = (z_dgs2 < -1.5) & (t10y2y_chg > 0)
        
        dovish_extreme = dovish_nlp | high_epu | dovish_market
        
        # 鹰派突变：文本极度变鹰 / 政策不确定性消除 / 短端利率暴涨且曲线变平
        hawkish_nlp = fomc_z < -2.0
        low_epu = epu_z < -2.0
        hawkish_market = (z_dgs2 > 1.5) & (t10y2y_chg < 0)
        
        hawkish_extreme = hawkish_nlp | low_epu | hawkish_market
        
        # 记录最近3日内是否发生过极端冲击
        recent_dovish = dovish_extreme.rolling(3).max().fillna(0) > 0
        recent_hawkish = hawkish_extreme.rolling(3).max().fillna(0) > 0
        
        # --- 反接飞刀衰竭条件 (Exhaustion) ---
        # 暴跌趋势衰竭：今日跌幅收窄或反抽 (即动量 > 5日均值)
        exhaustion_bull = dgs2_vel > dgs2_vel.rolling(5).mean()
        # 暴涨趋势衰竭：今日涨幅收窄或回落 (即动量 < 5日均值)
        exhaustion_bear = dgs2_vel < dgs2_vel.rolling(5).mean()
        
        # --- 信号组合与脉冲提取 (Sniper Pulse) ---
        # 同时满足：近期有极值 + 今日已衰竭
        dovish_signal = recent_dovish & exhaustion_bull
        hawkish_signal = recent_hawkish & exhaustion_bear
        
        # 边际脉冲：只在信号从 False 变为 True 的第一天触发
        dovish_pulse = dovish_signal & (~dovish_signal.shift(1).fillna(False))
        hawkish_pulse = hawkish_signal & (~hawkish_signal.shift(1).fillna(False))
        
        # 赋值 (美债是正Carry资产，鸽派利好TLT，鹰派利空TLT)
        signal.loc[dovish_pulse] = 1.0
        signal.loc[hawkish_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"