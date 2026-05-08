import numpy as np
import pandas as pd

class MicrostructurePanicExhaustionFactor:
    """微观恐慌交叉衰竭因子 (microstructure/nonlinear)

    逻辑: 将股票市场的微观抛售情绪(VIX)与影子银行系统的流动性微观结构(NFCI)进行非线性交叉。在系统性流动性危机中，微观层面的流动性枯竭与抛压冰冻往往并发。当两者均在极端高位形成共振，并且出现边际回落时，标志着微观层面的抛压彻底衰竭和流动性边际缓解，此时避险资本将回流长端美债(TLT)引发报复性反弹。必须等脉冲衰竭出现(二阶导数)才触发信号，防止在高危期接飞刀。
    数据: vixcls, nfci
    触发: 联合恐慌极值(VIX Z-Score + NFCI Z-Score > 3.5) + VIX回落(当日下跌且低于3日均值) + NFCI边际改善(最近5天内diff < 0)
    输出: 仅在联合极值衰竭的极短期内输出 +1.0 (看多美债脉冲)，非触发日常态输出 0.0。
    """

    def __init__(self):
        self.name = 'microstructure_panic_exhaustion_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，非触发日信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        
        # 计算 252 个交易日(1年)的 Z-Score 衡量水位的极端程度
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)
        
        # 铁律2: 二阶导数，严禁绝对值接飞刀，必须出现动能回落的衰竭迹象
        vix_exhaustion = (vix.diff() < 0) & (vix < vix.rolling(window=3).mean())
        
        if 'nfci' in data.columns:
            nfci = data['nfci'].ffill()
            nfci_mean = nfci.rolling(window=252, min_periods=60).mean()
            nfci_std = nfci.rolling(window=252, min_periods=60).std()
            nfci_z = (nfci - nfci_mean) / (nfci_std + 1e-6)
            
            # 铁律3: 边际变化，对于低频发布数据(NFCI为周频)，禁止使用绝对水平值判断当前变动
            # 使用 .diff() 捕捉边际恶化结束、拐头改善的瞬间，rolling(5) 确保公布发布周期的脉冲持续性
            nfci_diff = nfci.diff()
            nfci_exhaustion = nfci_diff.rolling(window=5, min_periods=1).min() < 0
            
            # 非线性特征交叉: VIX 与 NFCI 恐慌共振
            # 避免单指标硬性阈值导致触发率过低，采用组合 Z-score (负值截断，只考虑危机爆发即Z>0的叠加)
            combined_panic = vix_z.clip(lower=0) + nfci_z.clip(lower=0)
            
            # 双重极值 + 均出现边际衰竭
            trigger_condition = (combined_panic > 3.5) & vix_exhaustion & nfci_exhaustion
        else:
            # 兼容降级处理：如果没有 nfci 数据，使用单一高阈值的 VIX 衰竭反转
            trigger_condition = (vix_z > 2.5) & vix_exhaustion
            
        # 仅当触发狙击脉冲时输出 +1.0
        signal.loc[trigger_condition] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"