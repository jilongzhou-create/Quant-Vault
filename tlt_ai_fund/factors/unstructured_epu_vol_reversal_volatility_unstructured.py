import numpy as np
import pandas as pd

class UnstructuredEpuVolReversalFactor:
    """波动率极值与拥挤反转 (volatility/unstructured)

    逻辑: 结合基于新闻文本提取的经济政策不确定性(EPU)与跨资产恐慌波动率(美股VIX, 黄金GVZ)，构建非结构化综合恐慌指数。当不确定性和恐慌情绪处于极端高位且开始瓦解(二阶导数<0)时，表明流动性危机与恐慌抛售接近衰竭，避险资金转向美债，触发做多(TLT)脉冲；当市场极度自满且风险初露端倪时，触发做空脉冲。
    数据: usepuindxd (基于NLP的经济政策不确定性指数), vixcls (VIX), gvzcls (黄金VIX)
    触发: 综合恐慌 Z-Score > 1.2 且二阶衰竭 (动量回落) -> +1.0；Z-Score < -0.8 且风险抬头 -> -1.0
    输出: 零值休眠的狙击手级脉冲信号 [-1.0, 0.0, +1.0]
    """

    def __init__(self):
        self.name = 'unstructured_epu_vol_reversal'
        self.window = 252

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 1. 检查必要数据列是否存在
        required_cols = ['usepuindxd', 'vixcls', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 提取并前向填充，确保连续性
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 2. 边际变化与平滑：EPU 含有大量日度新闻噪音，取5日平滑以稳定二阶导数判断
        epu_smooth = epu.rolling(window=5, min_periods=1).mean()
        
        # 3. 计算滚动的 252 日 Z-Score
        def get_zscore(s, w):
            mean = s.rolling(window=w, min_periods=w//2).mean()
            std = s.rolling(window=w, min_periods=w//2).std()
            return (s - mean) / std.replace(0, 1e-5)
            
        epu_z = get_zscore(epu_smooth, self.window)
        vix_z = get_zscore(vix, self.window)
        gvz_z = get_zscore(gvz, self.window)
        
        # 综合跨资产恐慌指数 (Cross-Asset Panic Index) 
        # (因为多个Z-Score平均会降低方差，因此阈值可设置在1.2而非绝对的2.5)
        panic_idx = (epu_z.fillna(0) + vix_z.fillna(0) + gvz_z.fillna(0)) / 3.0
        
        # 4. 边际变化与二阶导数：计算3日动量差分，判断衰竭与反转
        panic_diff = panic_idx.diff(3)
        vix_diff = vix.diff(3)
        
        # 5. 产生脉冲信号 (遵循三大铁律)
        # 多头脉冲：处于恐慌极值 (水位高) AND 开始瓦解 (边际变化转负)
        long_cond = (panic_idx > 1.2) & (panic_diff < 0) & (vix_diff < 0)
        
        # 空头脉冲：处于自满极值 (水位低) AND 风险抬头 (边际变化转正)
        short_cond = (panic_idx < -0.8) & (panic_diff > 0) & (vix_diff > 0)
        
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"