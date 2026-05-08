import numpy as np
import pandas as pd

class RateCapitulationPivotFactor:
    """因子名称 (挖掘方向/方法): 政策转向与流动性冲量 / 非线性特征交叉

    逻辑: 捕捉市场对美联储政策预期发生剧烈反转的瞬间。当短端利率(2年期)出现极端回落(Z-Score<-1.75)、引发收益率曲线急速变陡(Bull Steepening)，且同步伴随美联储情感指数边际转鸽时，确认为有效的降息抢跑与流动性宽松脉冲，此时强烈看多美股。反之发生鹰派陡升则看空。
    数据: [dgs2, t10y2y, fomc_sentiment]
    输出: +1.0 强烈看多(鸽派抢跑), -1.0 强烈看空(鹰派反噬), 0.0 处于常态或非拐点
    触发条件: 2年期国债收益率5日动量极值 + 曲线5日变化显著 + FOMC情感跳升。脉冲信号，预期Trigger Rate约 5%-8%
    """

    def __init__(self):
        self.name = 'rate_capitulation_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 检查必需数据列
        req_cols = ['dgs2', 't10y2y', 'fomc_sentiment']
        if not all(col in data.columns for col in req_cols):
            return pd.Series(0.0, index=data.index, name=self.name)

        # 前向填充缺失值以处理节假日和低频阶梯数据
        df = data[req_cols].ffill()
        signal = pd.Series(0.0, index=df.index)

        # 2. 短端利率定价剧变 (捕捉抢跑降息或加息的极端冲量)
        # 计算2年期收益率的5日变化，并用126个交易日(约半年)进行Z-Score标准化，适应当前波动率Reigme
        dgs2_5d_chg = df['dgs2'].diff(5)
        dgs2_chg_mean = dgs2_5d_chg.rolling(window=126, min_periods=21).mean()
        dgs2_chg_std = dgs2_5d_chg.rolling(window=126, min_periods=21).std()
        dgs2_mom_z = (dgs2_5d_chg - dgs2_chg_mean) / (dgs2_chg_std + 1e-6)

        # 3. 收益率曲线动量 (确认是短端主导的Bull Steepening或Bear Flattening)
        t10y2y_5d_chg = df['t10y2y'].diff(5)

        # 4. FOMC边际态度跳变 (绝对禁止使用绝对值，只看3日内的Sentiment跃迁)
        fomc_sent_3d_chg = df['fomc_sentiment'].diff(3)

        # 5. 今日动量方向确认 (防接飞刀，确保今天的物理方向仍然符合脉冲)
        dgs2_1d_chg = df['dgs2'].diff(1)

        # ---------------- 核心非线性交叉逻辑 ----------------
        
        # 多头触发: 极度鸽派抢跑 (短端利率崩盘 + 曲线急陡 + 鸽派突变)
        bull_cond = (
            (dgs2_mom_z < -1.75) &           # 2Y收益率跌幅达到过去半年极值
            (t10y2y_5d_chg > 0.08) &         # 曲线在5天内急剧变陡(Bull Steepening) > 8 bps
            (fomc_sent_3d_chg > 0.10) &      # FOMC情绪边际显著转鸽
            (dgs2_1d_chg < 0)                # 当天短端利率仍在下行(恐慌/抢跑未衰竭反弹)
        )

        # 空头触发: 极度鹰派反噬 (短端利率飙升 + 曲线急平 + 鹰派突变)
        bear_cond = (
            (dgs2_mom_z > 1.75) &            # 2Y收益率升幅达到过去半年极值
            (t10y2y_5d_chg < -0.08) &        # 曲线在5天内急剧平坦化/倒挂 > 8 bps
            (fomc_sent_3d_chg < -0.10) &     # FOMC情绪边际显著转鹰
            (dgs2_1d_chg > 0)                # 当天短端利率仍在飙升
        )

        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0

        # ---------------- 零值休眠铁律强制处理 ----------------
        # 将连续触发的信号过滤，只保留信号突变的瞬间(Pulse)
        # 如果昨天和今天都是1.0，则今天强制归0，成为纯粹的狙击手信号
        signal = signal.where(signal != signal.shift(1), 0.0)
        
        # 填充可能由于滚动窗口导致的NaN
        signal = signal.fillna(0.0)
        signal.name = self.name

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"