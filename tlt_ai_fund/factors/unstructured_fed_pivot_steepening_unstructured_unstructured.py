import numpy as np
import pandas as pd

class UnstructuredFedPivotSteepeningFactor:
    """Fed Pivot 情绪冲击与曲线变陡脉冲因子 (unstructured/unstructured)

    逻辑: 捕捉美联储政策预期的极端跳跃 (意外转鸽/转鹰)。FOMC声明的文本情绪边际突变后，通过前瞻性最强的2年期收益率(dgs2)暴跌及曲线急剧牛陡(t10y2y)来印证降息预期的全面Price-in。为了不接主跌/主升浪飞刀，强制要求价格出现二阶导数衰竭(动量放缓)时才输出极值买卖脉冲。
    数据: fomc_sentiment (NLP政策情绪分数), dgs2 (2年期收益率), t10y2y (长短利差)
    触发: 
      看多脉冲 (+1.0): 过去10天FOMC大幅变鸽(Z > 2.0) AND (短端收益率暴跌达极值且跌势趋缓 OR 曲线牛陡达极值且变陡趋缓)
      看空脉冲 (-1.0): 过去10天FOMC大幅变鹰(Z < -2.0) AND (短端收益率暴涨达极值且涨势趋缓 OR 曲线熊平达极值且变平趋缓)
    输出: [-1.0, 1.0] 的离散脉冲信号
    """

    def __init__(self):
        self.name = 'unstructured_fed_pivot_steepening'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 要求必要的数据字段必须存在
        req_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        df = data[req_cols].ffill()
        signal = pd.Series(0.0, index=df.index)

        # =========================================================================
        # 1. 政策情绪跳跃 (Marginal Change Only 铁律)
        # =========================================================================
        # fomc_sentiment 是阶梯状低频变化数据，绝对禁止用绝对值，采用5日窗口捕捉会议日跳跃
        fomc_diff = df['fomc_sentiment'].diff(5)
        fomc_mean = fomc_diff.rolling(252).mean()
        fomc_std = fomc_diff.rolling(252).std().replace(0, np.nan).ffill().fillna(0.01)
        fomc_z = (fomc_diff - fomc_mean) / fomc_std
        
        # 捕捉事件发生后10天内的余波影响
        is_dove_shock = fomc_z.rolling(10).max() > 2.0
        is_hawk_shock = fomc_z.rolling(10).min() < -2.0

        # =========================================================================
        # 2. 短端利率暴跌/暴涨的交叉印证 (DGS2 Shock)
        # =========================================================================
        dgs2_diff = df['dgs2'].diff(5)
        dgs2_mean = dgs2_diff.rolling(252).mean()
        dgs2_std = dgs2_diff.rolling(252).std().replace(0, np.nan).ffill().fillna(0.01)
        dgs2_z = (dgs2_diff - dgs2_mean) / dgs2_std
        
        is_dgs2_plunge = dgs2_z < -2.0  # 收益率暴跌 = 看多美债印证
        is_dgs2_surge = dgs2_z > 2.0    # 收益率暴涨 = 看空美债印证
        
        # 二阶导数铁律: 防飞刀衰竭条件
        # 当短端收益率暴跌时，负的 diff 开始回升（大于其3日均值），说明跌速正在放缓
        dgs2_exhaustion_bull = dgs2_diff > dgs2_diff.rolling(3).mean()
        # 当短端收益率暴涨时，正的 diff 开始回落（小于其3日均值），说明涨速正在放缓
        dgs2_exhaustion_bear = dgs2_diff < dgs2_diff.rolling(3).mean()

        # =========================================================================
        # 3. 期限利差急剧变陡/变平的交叉印证 (T10Y2Y Shock)
        # =========================================================================
        t10y2y_diff = df['t10y2y'].diff(5)
        t10y2y_mean = t10y2y_diff.rolling(252).mean()
        t10y2y_std = t10y2y_diff.rolling(252).std().replace(0, np.nan).ffill().fillna(0.01)
        t10y2y_z = (t10y2y_diff - t10y2y_mean) / t10y2y_std
        
        is_bull_steepening = t10y2y_z > 2.0   # 降息预期导致短端暴跌更深 = 利差变大(牛陡)
        is_bear_flattening = t10y2y_z < -2.0  # 加息预期导致短端暴涨更猛 = 利差收窄(熊平)
        
        # 二阶导数铁律: 防飞刀衰竭条件
        steepening_exhaustion = t10y2y_diff < t10y2y_diff.rolling(3).mean()  # 变陡速度放缓
        flattening_exhaustion = t10y2y_diff > t10y2y_diff.rolling(3).mean()  # 变平速度趋缓

        # =========================================================================
        # 4. 信号合成: 情绪跳跃 AND (价格极值印证 + 二阶导数确认)
        # =========================================================================
        long_market_confirm = (is_dgs2_plunge & dgs2_exhaustion_bull) | (is_bull_steepening & steepening_exhaustion)
        short_market_confirm = (is_dgs2_surge & dgs2_exhaustion_bear) | (is_bear_flattening & flattening_exhaustion)

        long_cond = (is_dove_shock & long_market_confirm).fillna(False)
        short_cond = (is_hawk_shock & short_market_confirm).fillna(False)

        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"