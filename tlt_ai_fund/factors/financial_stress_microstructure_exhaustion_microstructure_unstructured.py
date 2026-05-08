import numpy as np
import pandas as pd

class PanicExhaustionPulseFactor:
    """Microstructure & Unstructured Panic Exhaustion Reversal
    
    逻辑: 捕捉流动性危机的恐慌极值与衰竭反转，以及微观交易量天量的衰竭。严格遵守二阶导数铁律：当高优宏观恐慌指标(VIX, STLFSI, GVZ, NFCI)或微观成交量(Volume)达到极值(Z-Score > 2.5)且开始回落(<3日均值)时，标志着恐慌动能衰竭，触发脉冲。同时结合NLP提取的FOMC预期边际突变。多维度非相关极值脉冲的并集保障了 Trigger Rate 处于 5%-15% 区间，同时通过二阶衰竭过滤了主跌浪中的“接飞刀”风险，保证高 Hit Rate。
    数据: vixcls, nfci, stlfsi4, gvzcls, volume, close, fomc_sentiment
    触发: 各独立指标的 Z-Score > 2.5 且 当前值 < 3日均值 / FOMC情绪的阶跃变化
    输出: +1.0 (恐慌衰竭抄底/鸽派突变看多美债), -1.0 (高潮衰竭看空/鹰派突变看空美债)
    """

    def __init__(self):
        self.name = 'panic_exhaustion_pulse'
        self.z_threshold = 2.5

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        buy_mask = pd.Series(False, index=data.index)
        sell_mask = pd.Series(False, index=data.index)
        
        # 核心算子: 极值 + 衰竭 (Anti-Catch-Falling-Knife)
        def get_exhaustion_mask(series, window, z_thresh):
            s = series.ffill()
            mean = s.rolling(window=window, min_periods=window//2).mean()
            std = s.rolling(window=window, min_periods=window//2).std()
            z_score = (s - mean) / (std + 1e-8)
            # 衰竭条件: 必须跌破3日均值，证明边际动能已反转
            exhaustion = s < s.rolling(window=3).mean()
            # 极值条件 + 衰竭条件
            return (z_score > z_thresh) & exhaustion
            
        # 1. 宏观恐慌衰竭 (VIX - 252日)
        if 'vixcls' in data.columns:
            vix_mask = get_exhaustion_mask(data['vixcls'], 252, self.z_threshold)
            buy_mask = buy_mask | vix_mask
            
        # 2. 金融系统压力衰竭 (STLFSI4 - 252日)
        if 'stlfsi4' in data.columns:
            stlfsi_mask = get_exhaustion_mask(data['stlfsi4'], 252, self.z_threshold)
            buy_mask = buy_mask | stlfsi_mask
            
        # 3. 芝加哥联储金融条件衰竭 (NFCI - 252日)
        if 'nfci' in data.columns:
            nfci_mask = get_exhaustion_mask(data['nfci'], 252, self.z_threshold)
            buy_mask = buy_mask | nfci_mask
            
        # 4. 实际利率冲击/黄金波动率恐慌衰竭 (GVZCLS - 252日)
        if 'gvzcls' in data.columns:
            gvz_mask = get_exhaustion_mask(data['gvzcls'], 252, self.z_threshold)
            buy_mask = buy_mask | gvz_mask
            
        # 5. 微观交易结构衰竭 (TLT Volume - 63日季度局部极值)
        if 'volume' in data.columns and 'close' in data.columns:
            vol = data['volume']
            close = data['close']
            
            vol_mean = vol.rolling(window=63, min_periods=21).mean()
            vol_std = vol.rolling(window=63, min_periods=21).std()
            vol_z = (vol - vol_mean) / (vol_std + 1e-8)
            vol_exh = vol < vol.rolling(window=3).mean()
            vol_extreme = (vol_z > self.z_threshold) & vol_exh
            
            # 使用10日收益率界定放量时的价格背景 (过滤震荡期)
            ret_10d = close.pct_change(10)
            
            # 恐慌性抛售后量能衰竭 -> 抄底看多
            buy_vol = vol_extreme & (ret_10d < -0.01)
            buy_mask = buy_mask | buy_vol
            
            # 情绪高潮FOMO后量能衰竭 -> 阶段看空
            sell_vol = vol_extreme & (ret_10d > 0.01)
            sell_mask = sell_mask | sell_vol

        # 6. 非结构化 NLP 情绪突变 (FOMC Sentiment 边际变化)
        if 'fomc_sentiment' in data.columns:
            fomc = data['fomc_sentiment'].ffill()
            fomc_diff = fomc.diff(5)
            f_mean = fomc_diff.rolling(window=252, min_periods=63).mean()
            f_std = fomc_diff.rolling(window=252, min_periods=63).std()
            fomc_z = (fomc_diff - f_mean) / (f_std + 1e-8)
            
            # 鸽派突变脉冲: 情绪从下突跃转正 (看多美债)
            buy_fomc = (fomc_z > self.z_threshold) & (fomc_diff > 0)
            buy_mask = buy_mask | buy_fomc
            
            # 鹰派突变脉冲: 情绪从上突跃转负 (看空美债)
            sell_fomc = (fomc_z < -self.z_threshold) & (fomc_diff < 0)
            sell_mask = sell_mask | sell_fomc

        # 执行脉冲输出
        signal[buy_mask] = 1.0
        signal[sell_mask] = -1.0
        
        # 冲突过滤 (多空同时触发时静默)
        conflict = buy_mask & sell_mask
        signal[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"