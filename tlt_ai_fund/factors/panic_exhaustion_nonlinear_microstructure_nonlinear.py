import numpy as np
import pandas as pd

class PanicExhaustionNonlinearPulseFactor:
    """Panic and Deflation Extreme Nonlinear Reversal (microstructure/nonlinear)

    逻辑: 恐慌与通缩预期的非线性极值衰竭反转。为了保证5%-15%的触发率，适度放宽了基础阈值，并通过将Z-Score平方进行非线性放大来捕捉真实的尾部拐点。当流动性危机引发的无差别抛售达到极值且开始边际衰竭时（VIX回落），避险买盘将推动美债上涨，触发脉冲做多；当通缩预期达到极值底且开始边际反弹时（原油反弹），再通胀预期重启导致长端利率上行，触发脉冲做空。
    数据: vixcls, dcoilwtico
    触发: VIX非线性Z-Score > 1.56 (即原始Z>1.25) 且 边际回落 -> +1.0; Oil非线性Z-Score < -1.56 且 边际反弹 -> -1.0
    输出: 狙击手级脉冲信号 [-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'panic_deflation_nonlinear_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 信号 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        req_cols = ['vixcls', 'dcoilwtico']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 前向填充缺失值 (如节假日)
        vix = data['vixcls'].ffill()
        oil = data['dcoilwtico'].ffill()
        
        # 计算 252 日滚动统计量 (约一年交易日)
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        oil_mean = oil.rolling(window=252, min_periods=60).mean()
        oil_std = oil.rolling(window=252, min_periods=60).std()
        oil_z = (oil - oil_mean) / (oil_std + 1e-8)
        
        # 特征非线性化 (Nonlinear Feature Transformation: 保留符号，放大极值)
        vix_z_nl = np.sign(vix_z) * (vix_z ** 2)
        oil_z_nl = np.sign(oil_z) * (oil_z ** 2)
        
        # 计算边际变化与均值 (二阶导数与边际变化铁律)
        vix_diff = vix.diff()
        oil_diff = oil.diff()
        vix_3d_mean = vix.rolling(window=3).mean()
        oil_3d_mean = oil.rolling(window=3).mean()
        
        # 触发条件 (1.25的平方为1.5625，代表约前10%的极端行情)
        # 做多条件: 恐慌极值 + 开始衰竭 (VIX低于3日均值且当日下降)
        long_cond = (vix_z_nl > 1.56) & (vix < vix_3d_mean) & (vix_diff < 0)
        
        # 做空条件: 通缩极值 + 开始反弹 (原油高于3日均值且当日上涨)
        short_cond = (oil_z_nl < -1.56) & (oil > oil_3d_mean) & (oil_diff > 0)
        
        # 赋值脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"