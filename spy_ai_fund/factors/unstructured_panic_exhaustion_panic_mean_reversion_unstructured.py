import numpy as np
import pandas as pd

class UnstructuredPanicExhaustionFactor:
    """非结构化新闻恐慌极值与衰竭脉冲 (panic_mean_reversion/unstructured)

    逻辑: 市场恐慌不仅体现在量价，更体现在非结构化新闻和会议纪要的情绪突变中。
          该因子使用基于新闻文本提取的美国经济政策不确定性指数(USEPUINDXD)和NLP提取的FOMC声明情绪(FOMC_SENTIMENT)。
          1. EPU衰竭看多：当新闻不确定性处于过去一年的相对高位(Z-Score > 1.2，极度恐慌)，且单日新闻不确定性剧烈回落(跌幅超过15%)时，意味着政策恐慌见顶退潮，风险溢价下降，产生强看多脉冲。
          2. EPU突发看空：当常态下突发黑天鹅(前日Z-Score平稳，今日单日飙升超40%)，产生突发避险看空脉冲。
          3. FOMC预期突变：使用阶梯状NLP数据的边际跳跃，FOMC情绪突发向鸽派/鹰派反转瞬间输出脉冲。
    数据: [usepuindxd, fomc_sentiment]
    输出: [-1.0, 1.0] 强看多为 +1.0, 突发避险看空为 -1.0, 常态为 0.0
    触发条件: 新闻EPU极值+回落 或 常态+暴涨 或 FOMC文本态度跳跃。由于需要极端变化率配合，整体预期Trigger Rate约 8%-12%。
    """

    def __init__(self, 
                 epu_z_threshold: float = 1.2, 
                 epu_drop_threshold: float = -0.15, 
                 epu_spike_threshold: float = 0.40, 
                 fomc_diff_threshold: float = 0.3):
        self.name = 'unstructured_panic_exhaustion_pulse'
        self.epu_z_threshold = epu_z_threshold
        self.epu_drop_threshold = epu_drop_threshold
        self.epu_spike_threshold = epu_spike_threshold
        self.fomc_diff_threshold = fomc_diff_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        has_epu = 'usepuindxd' in data.columns
        has_fomc = 'fomc_sentiment' in data.columns

        if not has_epu and not has_fomc:
            return signal
            
        # 1. 报纸政策不确定性指数 (非结构化文本情绪高频极值回落)
        if has_epu:
            epu = data['usepuindxd'].ffill()
            
            # 使用单日变化率捕捉情绪瞬间跳跃
            epu_ret = epu.pct_change()
            
            # 5日均线以衡量近期不确定性的基准水位，防止单日白噪声干扰极值判断
            epu_ma5 = epu.rolling(window=5).mean()
            
            # 252日 Z-score 衡量一年尺度下的系统性宏观恐慌水位
            epu_roll_mean = epu_ma5.rolling(window=252).mean()
            epu_roll_std = epu_ma5.rolling(window=252).std()
            epu_zscore = (epu_ma5 - epu_roll_mean) / (epu_roll_std + 1e-6)
            
            # 绝对铁律：二阶导数抄底法 (极度恐慌 + 衰竭)
            # 前一日的恐慌水位处于极值(>1.2)，且今日新闻不确定性断崖下跌
            long_cond = (epu_zscore.shift(1) > self.epu_z_threshold) & (epu_ret < self.epu_drop_threshold)
            
            # 短线避险：温和状态下突发黑天鹅暴涨
            # 绝对禁止在极度恐慌时看空(防止被主升浪轧空)，因此限定前一日 Z-Score 处于温和区间(<0.5)
            short_cond = (epu_zscore.shift(1) < 0.5) & (epu_ret > self.epu_spike_threshold)
            
            signal.loc[long_cond] = 1.0
            signal.loc[short_cond] = -1.0
            
        # 2. FOMC会议声明情绪文本得分 (低频非结构化数据边际跳跃)
        if has_fomc:
            fomc = data['fomc_sentiment'].ffill()
            
            # 绝对铁律：边际变化。FOMC Sentiment为前向填充的阶梯数据，仅捕捉变轨当天的跳跃
            fomc_diff = fomc.diff()
            
            fomc_long = fomc_diff > self.fomc_diff_threshold   # 明显转鸽
            fomc_short = fomc_diff < -self.fomc_diff_threshold # 明显转鹰
            
            signal.loc[fomc_long] = 1.0
            signal.loc[fomc_short] = -1.0
            
        # 确保只有极端时刻输出，过滤缺失值及假信号，常态休眠
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return (f"{self.__class__.__name__}(epu_z={self.epu_z_threshold}, "
                f"epu_drop={self.epu_drop_threshold}, epu_spike={self.epu_spike_threshold}, "
                f"fomc_diff={self.fomc_diff_threshold})")