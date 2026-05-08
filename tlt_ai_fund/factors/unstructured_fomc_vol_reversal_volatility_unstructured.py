import numpy as np
import pandas as pd

class UnstructuredFomcVolReversalFactor:
    """UnstructuredFomcVolReversalFactor (volatility/unstructured)

    逻辑: 结合 NLP 提取的 FOMC 情绪预期突变与跨资产恐慌波动率 (VIX/GVZ) 的衰竭确认。当美联储情绪发生极端边际跳跃（如超预期鸽派），且跨资产波动率停止狂飙并开始回落时，输出脉冲买入信号，绝对避免在波动率主跌浪中接飞刀。
    数据: fomc_sentiment (NLP情绪), vixcls (市场波动率), gvzcls (黄金跨资产波动率)
    触发: FOMC 情绪 5日动量变化 Z-Score 突破 1.5σ + 跨资产恐慌二阶衰竭 (VIX & GVZ 均小于 3日均值)。
    输出: 脉冲型 [-1.0, 1.0]。正值代表鸽派突变叠加恐慌消退(做多美债)，负值代表鹰派突变叠加波动消化(做空美债)，其余非触发日严格休眠为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_vol_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠, 初始信号严格全为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 检查必须的数据列
        req_cols = ['fomc_sentiment', 'vixcls']
        if not all(col in data.columns for col in req_cols):
            return signal

        fomc = data['fomc_sentiment'].ffill()
        vix = data['vixcls'].ffill()
        
        # 黄金波动率处理 (2008年前可能有缺失)
        gvz = data['gvzcls'].ffill() if 'gvzcls' in data.columns else pd.Series(np.nan, index=data.index)

        # ---------------------------------------------------------------------
        # 铁律3: 边际变化 (Marginal Change Only)
        # FOMC情绪是低频阶梯数据，使用 diff(5) 获取会议周前后的预期突变跳跃量。
        # 突变发生后，diff(5) 会在随后的 5 天内维持该跳跃差值，自然形成一个 5 天的"狙击狙击窗口"。
        # ---------------------------------------------------------------------
        fomc_5d_chg = fomc.diff(5).fillna(0.0)

        # 计算边际变化的 252 日滚动 Z-Score，只在"超预期"的震撼性跳跃时才捕捉
        fomc_chg_mean = fomc_5d_chg.rolling(window=252, min_periods=21).mean()
        fomc_chg_std = fomc_5d_chg.rolling(window=252, min_periods=21).std()
        fomc_chg_z = (fomc_5d_chg - fomc_chg_mean) / fomc_chg_std.replace(0, np.nan)

        # ---------------------------------------------------------------------
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 绝对禁止在高波动期直接入场！必须等待恐慌指标跌破近期(3日)均值，确认做空动能衰竭。
        # ---------------------------------------------------------------------
        vix_exhausting = vix < vix.rolling(window=3, min_periods=1).mean()

        # 跨资产确认 (黄金金价波动率同步回落 = 全局流动性恐慌消退)
        if gvz.notna().any():
            gvz_exhausting = (gvz < gvz.rolling(window=3, min_periods=1).mean()) | gvz.isna()
        else:
            gvz_exhausting = pd.Series(True, index=data.index)

        # 波动率全面衰竭信号
        vol_exhausting = vix_exhausting & gvz_exhausting

        # ---------------------------------------------------------------------
        # 信号合成区 (Sniper Pulse)
        # ---------------------------------------------------------------------
        
        # 做多脉冲: 情绪爆发式转鸽 (Z > +1.5σ) 且 水位偏鸽 + 跨资产恐慌回落
        dovish_pulse = (fomc_chg_z > 1.5) & (fomc > 0) & vol_exhausting

        # 做空脉冲: 情绪爆发式转鹰 (Z < -1.5σ) 且 水位偏鹰 + 恐慌回落 (市场平稳接受紧缩，长端上行)
        hawkish_pulse = (fomc_chg_z < -1.5) & (fomc < 0) & vol_exhausting

        # 赋值触发
        signal[dovish_pulse] = 1.0
        signal[hawkish_pulse & ~dovish_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"