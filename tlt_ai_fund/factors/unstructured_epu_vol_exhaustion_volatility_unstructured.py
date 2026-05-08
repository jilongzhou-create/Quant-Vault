import numpy as np
import pandas as pd

class UnstructuredPolicyRateVolReversalFactor:
    """Unstructured Policy & Rate Volatility Reversal (volatility/unstructured)

    逻辑: 结合非结构化新闻政策不确定性指数(EPU)与短端利率(2Y)的短期波动率。
          当宏观新闻不确定性与短端利率的联合波动率狂飙至极值，并开始衰竭时，标志着当前政策恐慌(不论是通胀恐慌还是衰退恐慌)见顶。
          为了区分恐慌的性质，引入短端利率的边际变化(10日动量)：
          - 若恐慌瓦解且短端利率暴跌(Bull Steepening)，说明市场正在定价紧急降息或避险，做多美债(+1.0)。
          - 若恐慌瓦解且短端利率飙升(Bear Flattening)，说明市场确认“Higher for Longer”通胀失控，做空美债(-1.0)。
          本逻辑彻底抛弃了全资产价格波动率(VIX等)，转而从文本新闻不确定性入手，正交性极佳。
    数据: usepuindxd (非结构化新闻政策不确定性), dgs2 (2年期美债收益率)
    触发: 联合波动率 252日 Z-Score > 1.25 + 波动率开始回落(二阶导数衰竭)
    输出: 狙击手脉冲信号，极端事件消退瞬间输出 +1.0 或 -1.0，常态输出 0.0
    """

    def __init__(self):
        self.name = 'unstructured_policy_rate_vol_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始信号全为0
        signal = pd.Series(0.0, index=data.index)
        
        # 必须数据缺失检查
        if 'usepuindxd' not in data.columns or 'dgs2' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        dgs2 = data['dgs2'].ffill()
        
        # 1. 计算联合波动率脉冲 (15个交易日约3周的动态波动率)
        epu_vol = epu.rolling(window=15, min_periods=5).std()
        dgs2_vol = dgs2.rolling(window=15, min_periods=5).std()
        
        # 计算 252日 长期分布 Z-Score
        epu_vol_mean = epu_vol.rolling(window=252, min_periods=21).mean()
        epu_vol_std = epu_vol.rolling(window=252, min_periods=21).std().replace(0, np.nan).fillna(1e-6)
        epu_vol_z = (epu_vol - epu_vol_mean) / epu_vol_std
        
        dgs2_vol_mean = dgs2_vol.rolling(window=252, min_periods=21).mean()
        dgs2_vol_std = dgs2_vol.rolling(window=252, min_periods=21).std().replace(0, np.nan).fillna(1e-6)
        dgs2_vol_z = (dgs2_vol - dgs2_vol_mean) / dgs2_vol_std
        
        # 等权构建宏观政策与利率的联合波动率
        combined_vol_z = (epu_vol_z.fillna(0) + dgs2_vol_z.fillna(0)) / 2.0
        
        # 2. 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 条件1：极值（Z > 1.25，捕捉约10.5%的尾部政策不确定性狂飙）
        is_extreme = combined_vol_z > 1.25
        # 条件2：衰竭（联合波动率开始从极值跌落，低于过去3天均值）
        is_exhausting = combined_vol_z < combined_vol_z.rolling(window=3).mean()
        
        # 3. 铁律3: 边际变化 (Marginal Change Only)
        # 用 10日短端利率变化来定性危机的方向，不看绝对水位
        dgs2_mom = dgs2.diff(10).fillna(0)
        
        # 逻辑合并：触发脉冲
        # 短端利率下行 > 5bps (降息预期发酵)
        bull_steepening = dgs2_mom < -0.05
        # 短端利率上行 > 5bps (通胀恶化，加息预期发酵)
        bear_flattening = dgs2_mom > 0.05
        
        trigger_long = is_extreme & is_exhausting & bull_steepening
        trigger_short = is_extreme & is_exhausting & bear_flattening
        
        signal[trigger_long] = 1.0
        signal[trigger_short] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"