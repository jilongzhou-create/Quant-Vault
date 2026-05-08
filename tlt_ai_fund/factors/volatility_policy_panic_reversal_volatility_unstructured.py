import numpy as np
import pandas as pd

class VolatilityEpuRegimeFactor:
    """波动率极值与EPU拥挤反转因子

    逻辑: 结合跨资产波动率(VIX)与非结构化经济政策不确定性(EPU)，捕捉市场恐慌/自满情绪的极值与衰竭。
         为了解决"波动率下降=避险退潮(看空)"还是"波动率下降=加息见顶(看多)"的胜率难题，
         通过收益率曲线(t10y2y)的20日动量变化严格区分宏观状态：
         - 若曲线变平(Hawkish/通胀冲击)，恐慌衰竭意味着加息预期见顶，做多美债；
         - 若曲线变陡(Dovish/衰退冲击)，恐慌衰竭意味着避险资金撤退，做空美债。
         反向自满唤醒逻辑同理，确保信号方向与宏观机制的因果匹配，提升条件IC。
    数据: vixcls, usepuindxd, t10y2y
    触发: (VIX_Z + EPU_Z) > 2.0 且回落 (恐慌衰竭) 或 < -1.5 且抬升 (自满打破脉冲)
    输出: 严格脉冲信号 [-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'vol_epu_regime_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必要字段
        required_cols = ['vixcls', 'usepuindxd', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 数据对齐与前向填充防缺失
        vix = data['vixcls'].ffill()
        epu = data['usepuindxd'].ffill()
        curve = data['t10y2y'].ffill()
        
        # 1. 波动率 Z-Score (252日)
        vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        
        # 2. EPU 具有高日频噪音，先进行 10日 平滑再算 Z-Score
        epu_ma = epu.rolling(10).mean()
        epu_z = (epu_ma - epu_ma.rolling(252).mean()) / epu_ma.rolling(252).std()
        
        # 填充缺失值并截断极端异常值，防止单一因子绝对主导
        vix_z = vix_z.fillna(0).clip(-3.0, 3.0)
        epu_z = epu_z.fillna(0).clip(-3.0, 3.0)
        
        # 构建综合宏观情绪压力指标 (Macro Stress)
        macro_stress = vix_z + epu_z
        
        # 3. 收益率曲线动量 (20日变化)，区分宏观冲击类型 
        # (平坦化<0 = 通胀/紧缩冲击，陡峭化>=0 = 衰退/宽松冲击)
        curve_mom = curve.diff(20).fillna(0)
        
        # 条件A: 极度恐慌且开始衰竭 (严格边际变化: 1日和3日均开始回落)
        is_exhausting = (
            (macro_stress > 2.0) & 
            (macro_stress.diff() < 0) & 
            (macro_stress.diff(3) < 0)
        )
        
        # 条件B: 极度自满且被打破 (严格边际变化: 1日和3日均开始抬升)
        is_awakening = (
            (macro_stress < -1.5) & 
            (macro_stress.diff() > 0) & 
            (macro_stress.diff(3) > 0)
        )
        
        # --- 信号生成与赋值 (遵守零值休眠铁律，仅在触发日赋值) ---
        
        # 恐慌衰竭 + 曲线平坦化 (如2022年通胀冲击尾声) -> 加息恐慌消退，做多美债
        signal.loc[is_exhausting & (curve_mom < 0)] = 1.0
        
        # 恐慌衰竭 + 曲线陡峭化 (如2020年流动性危机尾声) -> 避险资金撤退，做空美债
        signal.loc[is_exhausting & (curve_mom >= 0)] = -1.0
        
        # 自满打破 + 曲线平坦化 (如2022年初) -> 突发紧缩/通胀冲击，做空美债
        signal.loc[is_awakening & (curve_mom < 0)] = -1.0
        
        # 自满打破 + 曲线陡峭化 (如2020年初) -> 突发衰退冲击，资金涌入避险，做多美债
        signal.loc[is_awakening & (curve_mom >= 0)] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"