import numpy as np
import pandas as pd

class PolicyUncertaintyVolatilityPulseFactor:
    """政策不确定性与波动率衰竭脉冲因子 (panic_mean_reversion/nonlinear)

    逻辑: 
    美股是长牛+均值回归市场。当股市恐慌(VIX)处于极端高位且二阶导开始回落时，标志着恐慌衰竭，触发抄底看多脉冲(+1.0)；
    反之，当宏观政策不确定性发酵，且VIX在常态以上(0到1.5个标准差)温和连续攀升时，这代表钝刀割肉式的轻度恐慌恶化，输出防守看空脉冲(-1.0)。
    极端暴跌主浪期间输出0.0，坚决防范接飞刀。
    
    数据: vixcls (波动率), usepuindxd (经济政策不确定性指数)
    输出: 强烈看多(+1.0), 趋势看空(-1.0), 默认(0.0)
    触发条件: 极度恐慌见顶衰竭时(+1.0)，轻微恐慌叠加政策不确定性升温时(-1.0)，预期触发率在8%-12%左右。
    """

    def __init__(self):
        self.name = 'policy_uncertainty_volatility_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查依赖数据是否存在
        if 'vixcls' not in data.columns or 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)

        vix = data['vixcls'].ffill()
        epu = data['usepuindxd'].ffill()

        # 计算 VIX 252日均值和标准差，用于锚定恐慌的历史绝对极值 (Z-Score)
        vix_ma252 = vix.rolling(window=252, min_periods=60).mean()
        vix_std252 = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_ma252) / vix_std252

        # 计算 EPU 60日均线，用于判定政策不确定性所处的绝对水位
        epu_ma60 = epu.rolling(window=60, min_periods=20).mean()

        # --- 二阶导数衰竭条件 (防接飞刀) ---
        # 必须是今天比昨天回落，且跌穿最近3天均线，才被认定为"见顶回落"
        vix_exhaustion = (vix.diff(1) < 0) & (vix < vix.rolling(window=3).mean())

        # ====================================================
        # 买点 (抄底): 极度恐慌 + 动量衰竭
        # 规则: 仅在 VIX 突破 1.5 倍标准差历史极值，并且恐慌情绪开始降温的瞬间触发
        # ====================================================
        buy_cond = (vix_z > 1.5) & vix_exhaustion

        # ====================================================
        # 卖点 (防守/做空): 钝刀割肉的发酵期
        # 规则: VIX高于均值但尚未触发极端恐慌(Z_score在0到1.5之间) -> 属于阴跌市的典型特征
        # 同时短期内 VIX 显著升温(近3天上涨超1.0)，且政策不确定性高于均线并正在发酵
        # ====================================================
        sell_cond = (
            (vix_z >= 0.0) & (vix_z <= 1.5) &
            (vix.diff(1) > 0) & 
            (vix.diff(3) > 1.0) &
            (epu > epu_ma60) &
            (epu.diff(3) > 0)
        )

        # 组合脉冲信号输出
        signal = pd.Series(0.0, index=data.index)
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"