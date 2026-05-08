import numpy as np
import pandas as pd

class MacroPanicExhaustionFactor:
    """宏观恐慌衰竭因子 (volatility/nonlinear)

    逻辑: 流动性危机时所有资产(含美债)同跌(如2020年3月)，直接抄底美债会接飞刀。必须监控股市(VIX)与黄金(GVZ)等跨资产波动率极值同步见顶回落，结合政策不确定性(EPU)衰竭，捕捉无差别抛售结束、央行宽松预期发酵的脉冲买点。
    数据: vixcls, gvzcls, usepuindxd
    触发: 极端恐慌 (VIX Z-Score > 2.5 且开始回落) 叠加跨资产确认 (GVZ < 3日均值 且 diff < 0) -> +1.0
    输出: 脉冲型信号。极度恐慌衰竭时输出 +1.0 看多美债；极度自满且波动率重燃时输出 -1.0 看空美债。
    """

    def __init__(self):
        self.name = 'macro_panic_exhaustion_vol_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 字段检查
        required_cols = ['vixcls', 'gvzcls', 'usepuindxd']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 1. 基础数据处理 (处理节假日和缺失值)
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        epu = data['usepuindxd'].ffill()
        
        # 2. 核心水准指标计算 (滚动252日 Z-Score 防止前视偏差)
        vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        epu_z = (epu - epu.rolling(252).mean()) / epu.rolling(252).std()
        
        # 3. 衰竭/边际变化条件计算 (二阶导数防飞刀)
        vix_ma3 = vix.rolling(3).mean()
        gvz_ma3 = gvz.rolling(3).mean()
        
        vix_diff = vix.diff()
        gvz_diff = gvz.diff()
        
        # -------------------------------------------------------------
        # 条件A: 多头脉冲 (+1.0) - 恐慌衰竭 -> 流动性抛售结束 -> 美债避险功能回归
        # 极值条件: 股市恐慌 或 宏观政策不确定性 处于极值
        extreme_panic = (vix_z > 2.5) | (epu_z > 2.5)
        # 衰竭条件: VIX 与 GVZ 同步跌破3日均线，且当天边际回落
        panic_exhaustion = (vix < vix_ma3) & (vix_diff < 0) & (gvz < gvz_ma3) & (gvz_diff < 0)
        
        long_cond = extreme_panic & panic_exhaustion
        
        # -------------------------------------------------------------
        # 条件B: 空头脉冲 (-1.0) - 极度自满反转 -> 通胀/紧缩担忧重燃 -> 美债承压
        # 极值条件: 股市与政策不确定性双双处于极度乐观/自满水位
        extreme_complacency = (vix_z < -1.5) & (epu_z < -1.0)
        # 边际恶化: 跨资产波动率同步抬头突破3日均线
        complacency_reversal = (vix > vix_ma3) & (vix_diff > 0) & (gvz > gvz_ma3) & (gvz_diff > 0)
        
        short_cond = extreme_complacency & complacency_reversal
        
        # 4. 严格赋予脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"