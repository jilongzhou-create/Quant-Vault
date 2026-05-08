import numpy as np
import pandas as pd

class CrossAssetVolDivergenceFactor:
    """跨资产波动率分歧衰竭因子 (microstructure/options)

    逻辑: 股票隐含波动率(VIX)代表经济增长与股市恐慌, 黄金隐含波动率(GVZ)代表通胀失控与极端避险恐慌。
          当 GVZ 相对 VIX 出现极端向上偏离时(通胀/避险主导), 若恐慌开始衰竭, 意味着通胀或流动性危机见顶, 触发做多美债(脉冲)。
          当 VIX 相对 GVZ 出现极端向上偏离时(纯股市崩盘主导), 市场通常买入美债避险; 当该恐慌开始衰竭时, 资金回流风险资产(Risk-On), 触发做空美债(脉冲)。
    数据: vixcls (标普500波动率), gvzcls (黄金波动率)
    触发: 波动率相对偏离度的 63日(单季度) Z-Score 绝对值 > 1.5，且出现 3日均线拐点 (严格遵守二阶导数衰竭铁律)
    输出: 脉冲信号, 多空双向 [-1.0, 1.0], 常态休眠为 0.0
    """

    def __init__(self):
        self.name = 'cross_asset_vol_divergence'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 确保所需数据列存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 步骤1: 计算 63日(单季度) 波动率滚动基准
        vix_ma = vix.rolling(window=63, min_periods=21).mean()
        gvz_ma = gvz.rolling(window=63, min_periods=21).mean()
        
        # 步骤2: 计算相对波动率抬升度 (消除绝对数值的量纲差异)
        vix_elev = vix / (vix_ma + 1e-8)
        gvz_elev = gvz / (gvz_ma + 1e-8)
        
        # 步骤3: 计算跨资产波动率分歧度 (黄金恐慌 vs 股市恐慌)
        vol_div = gvz_elev - vix_elev
        
        # 步骤4: 计算分歧度的 63日 Z-Score (目标锁定 5%-15% 的极端脉冲区间)
        div_ma = vol_div.rolling(window=63, min_periods=21).mean()
        div_std = vol_div.rolling(window=63, min_periods=21).std()
        div_zscore = (vol_div - div_ma) / (div_std + 1e-8)
        
        # 步骤5: 二阶导数衰竭判定 (3日短均线拐点)
        div_3d_ma = vol_div.rolling(window=3, min_periods=1).mean()
        
        # 触发逻辑 A: 通胀/极端避险恐慌达到极值 + 开始衰竭 -> 收益率见顶回落 -> 做多美债 (+1.0)
        long_cond = (div_zscore > 1.5) & (vol_div < div_3d_ma)
        
        # 触发逻辑 B: 纯股市恐慌达到极值 + 开始衰竭 -> Risk-On 资金撤出避险国债 -> 做空美债 (-1.0)
        short_cond = (div_zscore < -1.5) & (vol_div > div_3d_ma)
        
        # 赋值狙击手脉冲信号 (默认 0.0)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"