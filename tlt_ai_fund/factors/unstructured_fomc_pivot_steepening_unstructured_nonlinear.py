import numpy as np
import pandas as pd

class UnstructuredFomcPivotSteepeningFactor:
    """FOMC情绪突变与曲线陡峭交叉因子 (unstructured/nonlinear)

    逻辑: 捕捉NLP提取的FOMC情绪的边际突变(鸽/鹰派转向)。为了过滤噪音和“嘴炮”，非线性交叉了收益率曲线的实际定价动作：只有当FOMC情绪显著转鸽，且情绪冲击动能开始衰竭(二阶导数<=0)时，叠加曲线出现真实的“牛陡”(Bull Steepening: 短端利率暴跌导致曲线快速变陡)，证明市场正在实质性Price-in降息，才输出做多美债脉冲。避免在未获资金面确认时接飞刀。
    数据: fomc_sentiment (FOMC情绪得分), t10y2y (10年-2年利差), dgs2 (2年期收益率)
    触发: FOMC情绪21日边际变化 Z-Score > 2.0 + 情绪变化停止加速(衰竭) + 曲线牛陡(T10Y2Y变陡 Z>1.5 且 DGS2下行 Z<-1.5) -> +1.0
    输出: 严格的狙击手级脉冲信号 [-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'unstructured_fomc_pivot_steepening_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，非触发日必须为 0.0
        signal = pd.Series(0.0, index=data.index)

        req_cols = ['fomc_sentiment', 't10y2y', 'dgs2']
        if not all(col in data.columns for col in req_cols):
            return signal

        # 提取数据并前向填充处理缺失值
        fomc = data['fomc_sentiment'].ffill()
        t10y2y = data['t10y2y'].ffill()
        dgs2 = data['dgs2'].ffill()

        # 铁律3: 边际变化 (Marginal Change Only)
        # 绝对禁止使用绝对值。FOMC情绪是低频阶梯数据，使用21日差分捕捉周期间的政策预期突变
        fomc_mom = fomc.diff(21)
        
        # 市场交易指标 t10y2y 和 dgs2 使用5日快速边际变化
        t10y2y_mom = t10y2y.diff(5)
        dgs2_mom = dgs2.diff(5)

        # 计算 Z-Score 以衡量极端程度 (禁止无意义魔法数字)
        # FOMC情绪变化较慢，使用252个交易日(约1年)作为分布基准
        fomc_mom_std = fomc_mom.rolling(252).std().replace(0, np.nan)
        fomc_mom_z = (fomc_mom - fomc_mom.rolling(252).mean()) / fomc_mom_std

        # 市场定价反应快，使用63个交易日(约1个季度)作为近期常态分布基准
        t10y2y_mom_std = t10y2y_mom.rolling(63).std().replace(0, np.nan)
        t10y2y_mom_z = (t10y2y_mom - t10y2y_mom.rolling(63).mean()) / t10y2y_mom_std

        dgs2_mom_std = dgs2_mom.rolling(63).std().replace(0, np.nan)
        dgs2_mom_z = (dgs2_mom - dgs2_mom.rolling(63).mean()) / dgs2_mom_std

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 动量停止上升/下降，代表预期冲击开始衰竭
        fomc_exhaustion_long = fomc_mom.diff(3) <= 0
        fomc_exhaustion_short = fomc_mom.diff(3) >= 0

        # 非线性特征交叉逻辑 (FICC经济学含义)
        # 多头 (做多美债): FOMC突变转鸽 (Z > 2.0) + 边际衰竭 + 市场确认牛陡 (利差快速走阔 Z>1.5 且 2年期利率极速下行 Z<-1.5)
        long_cond = (
            (fomc_mom_z > 2.0) & 
            fomc_exhaustion_long & 
            (t10y2y_mom_z > 1.5) & 
            (dgs2_mom_z < -1.5)
        )
        
        # 空头 (做空美债): FOMC突变转鹰 (Z < -2.0) + 边际衰竭 + 市场确认熊平/倒挂 (利差极速缩窄 Z<-1.5 且 2年期利率极速飙升 Z>1.5)
        short_cond = (
            (fomc_mom_z < -2.0) & 
            fomc_exhaustion_short & 
            (t10y2y_mom_z < -1.5) & 
            (dgs2_mom_z > 1.5)
        )

        # 生成脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"