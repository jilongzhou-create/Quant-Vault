import numpy as np
import pandas as pd

class UnstructuredFomcMomentumExhaustionFactor:
    """Unstructured FOMC Momentum Exhaustion (unstructured/unstructured)

    逻辑: 捕捉 FOMC 声明情绪(NLP)突变后的趋势确认脉冲。由于央行政策预期突变当天，美债市场往往经历高波动的"飞刀"行情，
          本因子不直接在突变当天输出信号，而是严格遵守边际变化与二阶导数铁律：提取情绪边际变化的动量(10日EMA的差分)，
          并在动量达到极值(Z-Score > 2.5)且开始产生二阶衰竭(回落至3日均线以下)时，才触发狙击手买卖脉冲。
          这代表市场对突变消息的初始恐慌/狂热已消化完毕，随后将展开顺势的安全波段。常态下严格保持 0 值休眠。
    数据: fomc_sentiment (基于LLM分析的FOMC文本鹰鸽情绪得分)
    触发: FOMC情绪边际动量的 252日 Z-Score > 2.5 (极值条件) 且 动量向均值回归即当前动量落后于3日均值 (衰竭条件)
    输出: 狙击手脉冲信号 [-1.0, 1.0]，非触发日严格为 0.0，看多美债=+1.0，看空美债=-1.0
    """

    def __init__(self):
        self.name = 'unstructured_fomc_momentum_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格遵守铁律1：零值休眠，初始化全为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 数据校验保护
        if 'fomc_sentiment' not in data.columns:
            return signal

        # 情绪得分是低频阶梯跳跃数据，非会议日使用前向填充保持当前状态
        fomc = data['fomc_sentiment'].ffill().fillna(0.0)

        # ---------------------------------------------------------------------
        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止直接使用 fomc_sentiment 绝对值。使用 10日 EMA 平滑阶梯，再进行一阶差分提取动量
        # ---------------------------------------------------------------------
        fomc_ema = fomc.ewm(span=10, adjust=False).mean()
        fomc_mom = fomc_ema.diff(1).fillna(0.0)

        # ---------------------------------------------------------------------
        # 极值条件评估 (Z-Score > 2.5)
        # 计算动量的 252个交易日 (约1年) 滚动 Z-Score
        # 加入 1e-8 极小值防止早期数据方差为0导致的除零报错
        # ---------------------------------------------------------------------
        mom_mean = fomc_mom.rolling(window=252, min_periods=21).mean()
        mom_std = fomc_mom.rolling(window=252, min_periods=21).std() + 1e-8
        mom_zscore = (fomc_mom - mom_mean) / mom_std

        # ---------------------------------------------------------------------
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 计算动量的 3日均线，要求动量必须从极值点开始衰竭回归，过滤掉最危险的主跌/主升浪
        # ---------------------------------------------------------------------
        mom_ma3 = fomc_mom.rolling(window=3, min_periods=1).mean()

        # 鸽派突变 (利好美债 TLT)
        # 1. 动量极值飙升 (Z > 2.5)
        # 2. 动量开始衰竭 (当前多头动量 < 近3日均线)
        dove_shock = (mom_zscore > 2.5)
        dove_exhaustion = (fomc_mom < mom_ma3)

        # 鹰派突变 (利空美债 TLT)
        # 1. 负向动量极值飙升 (Z < -2.5)
        # 2. 负向动量开始衰竭 (当前空头动量向上收缩 > 近3日均线)
        hawk_shock = (mom_zscore < -2.5)
        hawk_exhaustion = (fomc_mom > mom_ma3)

        # 只在极值 + 衰竭同时满足的极少数几日内触发脉冲
        signal[dove_shock & dove_exhaustion] = 1.0
        signal[hawk_shock & hawk_exhaustion] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"