import numpy as np
import pandas as pd

class PanicExhaustionReversalFactor:
    """Cross-Asset Panic Exhaustion Reversal (microstructure/nonlinear)

    逻辑: 跨资产恐慌极值与衰竭反转。当股票(VIX)和避险资产(GVZ)的波动率同步飙升至极端水平，
          随后动量开始衰竭回落时，标志着流动性危机(Dash-for-cash)的抛售见顶。若此时2年期美债收益率(DGS2)同步下行，
          说明宏观面已确认并计入美联储降息宽松预期，做多TLT；
          反之，若波动率从极度自满中苏醒且短端利率上行，说明遭遇紧缩/通胀冲击，做空TLT。
          此设计严格遵循狙击手脉冲铁律，确保在恐慌出尽、拐点确立的瞬间捕捉交易脉冲，避免接飞刀。
    数据: vixcls, gvzcls, dgs2
    触发: 波动率综合Z-Score(126日) > 1.5 且跌破3日均线并出现负增量 + DGS2 3日边际下行 -> +1.0
          波动率综合Z-Score(126日) < -1.5 且突破3日均线并出现正增量 + DGS2 3日边际上行 -> -1.0
    输出: 脉冲型信号，[-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'cross_asset_panic_exhaustion_reversal_microstructure_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须数据列检查
        required_cols = ['vixcls', 'dgs2']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 数据填充，防止缺失值导致的断层
        vix = data['vixcls'].ffill()
        dgs2 = data['dgs2'].ffill()
        # GVZ 缺失时使用 VIX 作为替代，保持维度一致性
        gvz = data['gvzcls'].ffill() if 'gvzcls' in data.columns else vix
        
        # 126交易日(约半年)滚动窗口，符合 FICC 宏观资金的中期调仓观测周期
        window = 126
        
        # 计算 VIX Z-Score
        vix_mean = vix.rolling(window=window).mean()
        vix_std = vix.rolling(window=window).std().replace(0, 1e-5)
        vix_z = (vix - vix_mean) / vix_std
        
        # 计算 GVZ Z-Score (黄金波动率，代表避险资产的流动性压力)
        gvz_mean = gvz.rolling(window=window).mean()
        gvz_std = gvz.rolling(window=window).std().replace(0, 1e-5)
        gvz_z = (gvz - gvz_mean) / gvz_std
        
        # 构建非线性交叉的综合恐慌指数 
        # (双资产Z-Score相加，Z > 1.5 约代表整体处于Top 15%左右的联合尾部极端区间)
        panic_idx = vix_z + gvz_z
        
        # 铁律2: 二阶导数与衰竭条件 (绝对禁止单看绝对值)
        panic_ma3 = panic_idx.rolling(window=3).mean()
        panic_diff = panic_idx.diff()
        
        # 恐慌衰竭 (极值后跌破均线且动量向下)
        panic_exhausting = (panic_idx < panic_ma3) & (panic_diff < 0)
        
        # 自满苏醒 (底部突破均线且动量向上)
        panic_waking = (panic_idx > panic_ma3) & (panic_diff > 0)
        
        # 铁律3: 边际变化 (结合 DGS2 短端利率动量确认)
        dgs2_diff3 = dgs2.diff(3)
        
        # 触发条件: 
        # 多头脉冲: 综合恐慌指数 > 1.5 (极端恐慌) + 恐慌见顶衰竭 + 短端利率边际下行(Pivot确认)
        bull_cond = (panic_idx > 1.5) & panic_exhausting & (dgs2_diff3 < 0)
        
        # 空头脉冲: 综合恐慌指数 < -1.5 (极端自满) + 波动率开始苏醒 + 短端利率边际上行(紧缩冲击)
        bear_cond = (panic_idx < -1.5) & panic_waking & (dgs2_diff3 > 0)
        
        # 赋值脉冲信号 (使用 fillna(False) 避免 NaN 造成的索引错误)
        signal.loc[bull_cond.fillna(False)] = 1.0
        signal.loc[bear_cond.fillna(False)] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"