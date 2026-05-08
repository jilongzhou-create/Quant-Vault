import numpy as np
import pandas as pd

class MicrostructureLiquidityExhaustionNonlinearFactor:
    """微观流动性恐慌衰竭与反转因子 (microstructure/nonlinear)

    逻辑: 结合高频跨资产恐慌(VIX)与低频系统性金融压力(NFCI/STLFSI4)的非线性交叉。
          TLT 是典型的避险与流动性资产。当流动性危机爆发时，短期的无差别抛售会导致TLT也被错杀；
          一旦极度恐慌见顶且开始呈现边际衰竭，流动性重新释放，避险回归与降息预期将推动美债大幅反弹。
          必须是脉冲信号：仅在"极端高压 + 开始回落"的瞬间转折窗口触发，坚决避开危机主跌浪的飞刀期。
          反之，当市场处于极度宽松且开始边际恶化时，预示强劲复苏及紧缩交易将起，发出看空脉冲。
    数据: vixcls, nfci (缺失时回退至 stlfsi4)
    触发: VIX Z-Score > 2.0 且边际回落 + 金融压力 Z-Score > 1.5 且边际回落 -> +1.0
          VIX Z-Score < -1.5 且边际回升 + 金融压力 Z-Score < -1.5 且边际回升 -> -1.0
    输出: [-1.0, 1.0] 的多空狙击手脉冲信号，常态下保持 0.0 零值休眠。
    """

    def __init__(self):
        self.name = 'microstructure_liquidity_exhaustion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 初始信号全为 0.0，只在特定脉冲日触发
        signal = pd.Series(0.0, index=data.index)
        
        # 确保必要数据存在，高价值字段互为备份
        if 'vixcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        
        if 'nfci' in data.columns:
            stress_idx = data['nfci'].ffill()
        elif 'stlfsi4' in data.columns:
            stress_idx = data['stlfsi4'].ffill()
        else:
            return signal
            
        # 1. 宏观极端水位特征: 252日(约1交易年)滚动 Z-Score 评估绝对极值
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean) / vix_std
        
        stress_mean = stress_idx.rolling(window=252, min_periods=60).mean()
        stress_std = stress_idx.rolling(window=252, min_periods=60).std()
        stress_z = (stress_idx - stress_mean) / stress_std
        
        # 2. 衰竭特征 (铁律2 & 铁律3: 二阶导数与边际变化)
        # VIX 为高频日度数据，使用 3 日均值捕捉快速的恐慌/狂热反转
        vix_exhaustion_long = vix < vix.rolling(window=3).mean()
        vix_exhaustion_short = vix > vix.rolling(window=3).mean()
        
        # 金融压力指数(NFCI/STLFSI4)通常为周频发布并插值
        # 铁律3: 绝对禁止对阶梯状数据直接用绝对值！使用 5 日(1周)均值比较捕捉其台阶式边际改善/恶化
        stress_exhaustion_long = stress_idx < stress_idx.rolling(window=5).mean()
        stress_exhaustion_short = stress_idx > stress_idx.rolling(window=5).mean()
        
        # 3. 脉冲触发条件: 非线性特征交叉
        # 多头脉冲: 双重恐慌极值同步开始缓解 (防飞刀抄底)
        long_cond = (
            (vix_z > 2.0) & vix_exhaustion_long & 
            (stress_z > 1.5) & stress_exhaustion_long
        )
        
        # 空头脉冲: 双重环境极度宽松并同步边际反转向上 (逃顶并做空)
        short_cond = (
            (vix_z < -1.5) & vix_exhaustion_short & 
            (stress_z < -1.5) & stress_exhaustion_short
        )
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"