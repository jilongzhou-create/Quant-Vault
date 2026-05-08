import numpy as np
import pandas as pd

class DollarSqueezePanicExhaustionFactor:
    """美元流动性挤兑与恐慌衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合美股VIX与广义贸易加权美元指数(DTWEXBGS)。强宏观避险时，全球资金抛售资产换取美元(流动性挤兑)，导致美元升值且VIX飙升。当VIX处高位且两者同时转跌回落时，标志流动性恐慌衰竭，触发看多抄底脉冲(防接飞刀)；当VIX处于常态未极化时，若短期内两者同时暴涨，则确认为宏观避险发酵，触发看空脉冲。
    数据: vixcls (VIX), dtwexbgs (贸易加权美元指数)
    输出: 强看多(+1.0)或强看空(-1.0)，其他常态时间输出 0.0
    触发条件: 狙击手级别的瞬时脉冲。多头: VIX_Z>1, USD_Z>0, 且单日二阶导双双转负。空头: VIX_Z (0~1.5)且3日动量大幅齐升。预期 Trigger Rate: 5%-15%
    """

    def __init__(self):
        self.name = 'dollar_squeeze_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖列是否存在
        if 'vixcls' not in data.columns or 'dtwexbgs' not in data.columns:
            return signal
            
        # 前向填充填补低频/节假日缺失值
        vix = data['vixcls'].ffill()
        usd = data['dtwexbgs'].ffill()
        
        # 252日均值回归 Z-Score，识别是否处于历史极端状态
        vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        usd_z = (usd - usd.rolling(252).mean()) / usd.rolling(252).std()
        
        # 边际变化(动量与二阶导数)，识别爆发与衰竭时刻
        vix_diff_1 = vix.diff(1)
        usd_diff_1 = usd.diff(1)
        
        vix_pct_3 = vix.pct_change(3).replace([np.inf, -np.inf], 0)
        usd_pct_3 = usd.pct_change(3).replace([np.inf, -np.inf], 0)
        
        # 【抄底逻辑：防接飞刀 - 极值+衰竭】
        # VIX 处于较高位 (Z > 1.0)，美元处于相对强势区 (Z > 0.0)
        # 且当日两者双双下跌 (动能正式由正转负，避险情绪与流动性紧张同步衰竭)
        long_base = (
            (vix_z > 1.0) & 
            (usd_z > 0.0) & 
            (vix_diff_1 < 0) & 
            (usd_diff_1 < 0)
        )
        # 提取从不满足到满足瞬间的脉冲
        long_pulse = long_base & (~long_base.shift(1).fillna(False))
        
        # 【看空逻辑：钝刀割肉防逼空 - 常态恶化初期】
        # 此时波动率刚开始抬头，尚未到达能抄底的极值 (0.0 < Z <= 1.5)
        # VIX 3天急剧飙升 > 10%, 美元3天同步升值 > 0.5% (确认不是单一资产波动，而是宏观流动性抽离)
        # 且当日VIX依然在创新高，尚未衰竭
        short_base = (
            (vix_z > 0.0) & 
            (vix_z <= 1.5) & 
            (vix_pct_3 > 0.10) & 
            (usd_pct_3 > 0.005) & 
            (vix_diff_1 > 0)
        )
        # 提取恶化瞬间的脉冲
        short_pulse = short_base & (~short_base.shift(1).fillna(False))
        
        # 信号赋值
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"