import numpy as np
import pandas as pd

class CrossAssetLiquidityMicrostructureFactor:
    """跨资产流动性微观结构非线性反转因子 (microstructure/nonlinear)

    逻辑: 结合美股波动率(VIX)、黄金波动率(GVZ)与无风险流动性(dtb3)的非线性特征。跨资产波动率同步极值代表真正的全市场流动性枯竭(无差别抛售)。必须等恐慌见顶衰竭且前端利率开始下行(流动性注入/抢筹)才触发脉冲，避免在主跌浪接飞刀。反之在极端自满且流动性收紧时做空。
    数据: vixcls, gvzcls, dtb3
    触发: VIX Z-Score > 2.0 且 GVZ Z-Score > 1.5，叠加两者均低于3日均值(恐慌衰竭)，同时 dtb3的3日边际变化 < 0 -> 输出 +1.0
    输出: [-1.0, 1.0] 的狙击手脉冲信号
    """

    def __init__(self):
        self.name = 'cross_asset_liquidity_microstructure'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'gvzcls', 'dtb3']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return signal
            
        df = data[required_cols].ffill()
        
        # 计算长周期(252个交易日，约一年)的标准化极值
        vix_z = (df['vixcls'] - df['vixcls'].rolling(252).mean()) / df['vixcls'].rolling(252).std()
        gvz_z = (df['gvzcls'] - df['gvzcls'].rolling(252).mean()) / df['gvzcls'].rolling(252).std()
        
        # 铁律2: 二阶导数条件 (防接飞刀，必须等恐慌或自满开始反转衰竭)
        vix_ma3 = df['vixcls'].rolling(3).mean()
        gvz_ma3 = df['gvzcls'].rolling(3).mean()
        
        vix_exhausting = df['vixcls'] < vix_ma3
        gvz_exhausting = df['gvzcls'] < gvz_ma3
        
        vix_bouncing = df['vixcls'] > vix_ma3
        gvz_bouncing = df['gvzcls'] > gvz_ma3
        
        # 铁律3: 边际变化条件 (前端利率的微观动量变化，避免使用绝对水位)
        dtb3_diff = df['dtb3'].diff(3)
        
        # 非线性交叉做多条件: 多重恐慌同步极值 + 抛售见顶衰竭 + 短端利率急跌(微观流动性好转/降息定价启动)
        long_cond = (
            (vix_z > 2.0) & 
            (gvz_z > 1.5) & 
            vix_exhausting & 
            gvz_exhausting & 
            (dtb3_diff < 0)
        )
        
        # 非线性交叉做空条件: 多重自满极值 + 波动反弹 + 短端利率上行(微观流动性边际收紧)
        short_cond = (
            (vix_z < -1.5) & 
            (gvz_z < -1.5) & 
            vix_bouncing & 
            gvz_bouncing & 
            (dtb3_diff > 0)
        )
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"