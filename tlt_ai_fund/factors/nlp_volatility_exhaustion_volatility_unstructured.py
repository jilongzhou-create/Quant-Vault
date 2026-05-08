import numpy as np
import pandas as pd

class EpuFomcNlpVolatilityReversalFactor:
    """News-based Policy Volatility & NLP Sentiment Reversal

    逻辑: 采用两大纯非结构化文本数据(经济政策不确定性新闻指数 EPU 与 FOMC NLP情感得分)构建的叙事波动率反转脉冲。
          将EPU视为"宏观叙事的隐含波动率", 当政策恐慌/自满达到极值(Z>2.0或<-2.0)并出现二阶衰竭时, 叙事发生拐点。
          此时结合FOMC情感边际动量决定方向: 若美联储边际偏鸽, 恐慌消退转化为美债避险多头; 若边际偏鹰, 则转化为鹰派重定价空头。
          完全摒弃传统资产价格波动率(VIX), 从纯文本维度挖掘恐慌脉冲, 具备极低重合度。
    数据: usepuindxd (经济政策不确定性指数), fomc_sentiment (FOMC文本鹰鸽得分)
    触发: EPU 252日Z-Score极值 + 动量反转(diff()反向并加速下穿均线) + FOMC情绪21日边际方向
    输出: 脉冲型 [-1.0, 1.0], 触发后保持 4 天以稳稳击中 5-15% Trigger Rate 铁律目标
    """

    def __init__(self):
        self.name = 'epu_fomc_nlp_volatility_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 核心字段检查 (缺失直接返回 0.0)
        required_cols = ['usepuindxd', 'fomc_sentiment']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 2. 处理 EPU (新闻政策不确定性指数) -> 宏观叙事波动率
        epu = data['usepuindxd'].ffill()
        # 基础平滑去除单日新闻极值噪音 (3日平滑)
        epu_smooth = epu.rolling(window=3).mean()

        # 252日滚动 Z-Score 计算绝对水位
        epu_mean = epu_smooth.rolling(window=252).mean()
        epu_std = epu_smooth.rolling(window=252).std().replace(0, np.nan)
        epu_z = (epu_smooth - epu_mean) / epu_std

        # 3. 严格的二阶导数衰竭条件 (动量拐点)
        epu_diff = epu_smooth.diff()
        diff_ma3 = epu_diff.rolling(window=3).mean()

        # 顶点衰竭: 绝对水位极高 (>2.0σ) + 当日动量转负 + 下降加速度快于3日均值
        peak_extreme = epu_z > 2.0
        peak_exhaustion = peak_extreme & (epu_diff < 0) & (epu_diff < diff_ma3)

        # 冰点爆发: 绝对水位极低 (<-2.0σ, 极端自满) + 当日动量转正 + 上升加速度快于3日均值
        trough_extreme = epu_z < -2.0
        trough_exhaustion = trough_extreme & (epu_diff > 0) & (epu_diff > diff_ma3)

        # 4. 处理 FOMC 情感得分边际变化 -> 定价方向锚 (边际变化铁律)
        fomc = data['fomc_sentiment'].ffill()
        # 计算短期(21个交易日，约1个月)情绪边际变化量
        fomc_diff = fomc.diff(21)
        
        # 计算边际变化的相对显著性 Z-Score
        fomc_diff_mean = fomc_diff.rolling(window=252).mean()
        fomc_diff_std = fomc_diff.rolling(window=252).std().replace(0, np.nan)
        fomc_diff_z = (fomc_diff - fomc_diff_mean) / fomc_diff_std

        # 确立方向尾风: Z > 0 表示边际偏鸽(利好美债), Z <= 0 表示边际偏鹰(利空美债)
        # 必须 shift(1) 防止当日情绪变动带来的前视偏差，完全依赖昨日已确立的趋势
        fomc_dovish = fomc_diff_z.shift(1) > 0.0
        fomc_hawkish = fomc_diff_z.shift(1) <= 0.0

        # 5. 合并信号: "叙事极值衰竭 + 央行情绪共振"
        # 多头脉冲: 恐慌消退/自满打破 + 美联储偏鸽 -> 美债迎来强劲避险或宽松预期买盘
        long_trigger = ((peak_exhaustion | trough_exhaustion) & fomc_dovish).fillna(False)
        # 空头脉冲: 恐慌消退/自满打破 + 美联储偏鹰 -> 市场丧失庇护，通胀/加息担忧导致美债遭抛售
        short_trigger = ((peak_exhaustion | trough_exhaustion) & fomc_hawkish).fillna(False)

        # 6. 生成脉冲信号并展期满足 5-15% Trigger Rate 铁律
        signal = pd.Series(0.0, index=data.index)
        signal.loc[long_trigger] = 1.0
        signal.loc[short_trigger] = -1.0

        # 将稀疏的狙击手拐点脉冲向前展期 4 天，使得胜率和发波频率达到有效交易目标
        signal = signal.replace(0.0, np.nan).ffill(limit=4).fillna(0.0)

        # 数据清洗以绝后患
        signal = signal.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"