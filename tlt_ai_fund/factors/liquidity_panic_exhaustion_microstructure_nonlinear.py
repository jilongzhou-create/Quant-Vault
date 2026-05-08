import numpy as np
import pandas as pd

class LiquidityPanicExhaustionFactor:
    """流动性恐慌极值与衰竭反转 (microstructure/nonlinear)

    逻辑: 捕捉金融系统微观流动性压力(STLFSI4)与跨资产恐慌(VIX)的极致点。绝对禁止在恐慌飙升期接飞刀，只有当系统性压力指标(STLFSI4)处于极端高位，且恐慌情绪(VIX)开始实质性回落(二阶导数衰竭)时，才确认流动性挤兑结束。此时无差别抛售完毕，避险资金重返美债，触发强烈的脉冲式做多信号。
    数据: vixcls, stlfsi4
    触发: STLFSI4 Z-Score > 2.0 或 VIX Z-Score > 2.5 (极值)，且 VIX 跌破3日均值并单日下跌(边际衰竭)，且 STLFSI4 未再恶化。反之在极度自满且波动率反转抬升瞬间输出看空脉冲。
    输出: 狙击手级别的脉冲信号，触发衰竭时输出 +1.0 或 -1.0，其余非极端状态严格休眠返回 0.0。
    """

    def __init__(self):
        self.name = 'liquidity_panic_exhaustion_microstructure_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须处理数据缺失，避免报错
        if 'vixcls' not in data.columns or 'stlfsi4' not in data.columns:
            return signal
            
        # 采用前向填充处理低频和节假日缺失数据
        vix = data['vixcls'].ffill()
        fsi = data['stlfsi4'].ffill()
        
        # 1. 宏观极值计算：采用 252 个交易日（约一年）捕捉宏观周期级别的极值
        vix_mean = vix.rolling(window=252, min_periods=63).mean()
        vix_std = vix.rolling(window=252, min_periods=63).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        fsi_mean = fsi.rolling(window=252, min_periods=63).mean()
        fsi_std = fsi.rolling(window=252, min_periods=63).std()
        fsi_z = (fsi - fsi_mean) / (fsi_std + 1e-8)
        
        # 2. 边际变化与二阶导数衰竭条件 (Anti-Catch-Falling-Knife)
        # VIX 明确衰竭：必须低于过去3日均值，且当天发生实质性回落 (动量翻负)
        vix_exhaustion = (vix < vix.rolling(window=3).mean()) & (vix.diff() < 0)
        
        # FSI 企稳：由于压力指数具有低频阶梯变化特性，要求其不高于过去5日均值(停止恶化)
        fsi_stable = fsi <= fsi.rolling(window=5).mean()
        
        # 3. 非线性交叉触发逻辑
        # 看多条件：宏观金融压力或跨资产波动率达到极端水平
        extreme_panic = (fsi_z > 2.0) | (vix_z > 2.5)
        
        # 脉冲点：极值高位 + 开始衰竭回落 -> 流动性危机解压，避险资金买入美债
        long_trigger = extreme_panic & vix_exhaustion & fsi_stable
        
        # 看空条件：从极度自满状态中反转
        # 极度自满(两者均处于一年来的历史低位 Z < -1.5)
        extreme_complacency = (fsi_z < -1.5) & (vix_z < -1.5)
        # 波动率骤然抬头(突破3日均值且单日上涨) -> 抛售风险资产初期，现金为王压制美债
        vix_reversal = (vix > vix.rolling(window=3).mean()) & (vix.diff() > 0)
        fsi_rising = fsi >= fsi.rolling(window=5).mean()
        
        short_trigger = extreme_complacency & vix_reversal & fsi_rising
        
        # 4. 严格赋予脉冲信号
        signal[long_trigger] = 1.0
        signal[short_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"