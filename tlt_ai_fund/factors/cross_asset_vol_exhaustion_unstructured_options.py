import numpy as np
import pandas as pd

class CrossAssetPanicExhaustionFactor:
    """跨资产恐慌衰竭脉冲 (unstructured/options)

    逻辑: 
    1. VIX与GVZ(黄金ETF波动率)差值极端扩大(股票恐慌远超黄金恐慌, 典型流动性/衰退冲击)。当该差值见顶回落时, 代表流动性无差别抛售衰竭, 美联储通常介入, 美债迎来修复性买入。
    2. VIX与GVZ差值极端缩小(黄金波动率飙升, 典型实际利率/紧缩冲击)。当该差值见底回升时, 代表紧缩恐慌见顶, 收益率往往见顶回落, 利多美债。
    3. GVZ极端低迷后开始抬头, 代表市场对通胀/利率的极度自满(Complacency)被打破, 利率上行风险加剧, 提供做空美债的狙击点。
    
    数据: vixcls (VIX), gvzcls (黄金ETF隐含波动率)
    触发: 
    - 恐慌衰竭做多: (VIX-GVZ) 126日Z-Score > 1.7 且跌破3日均值; 或 Z-Score < -1.7 且突破3日均值。
    - 自满打破做空: GVZ 126日Z-Score < -1.7 且突破3日均值。
    输出: [-1.0, 1.0] 极端脉冲信号, Target Trigger Rate 5-15%
    """

    def __init__(self):
        self.name = 'cross_asset_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须严格为 0.0 (脉冲铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖列
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算 VIX 和 GVZ 的差值 (衡量跨资产恐慌的分化程度)
        spread = vix - gvz
        
        # 使用 126个交易日 (约半年) 计算敏捷的局部 Z-Score
        window = 126
        
        # Spread 的 Z-Score 计算
        spread_mean = spread.rolling(window=window, min_periods=window//2).mean()
        spread_std = spread.rolling(window=window, min_periods=window//2).std()
        spread_std = spread_std.replace(0, np.nan)
        spread_z = (spread - spread_mean) / spread_std
        
        # Spread 边际变化衰竭确认 (3日均线)
        spread_ma3 = spread.rolling(window=3, min_periods=1).mean()
        
        # GVZ 的 Z-Score 计算 (用于捕捉利率自满情绪)
        gvz_mean = gvz.rolling(window=window, min_periods=window//2).mean()
        gvz_std = gvz.rolling(window=window, min_periods=window//2).std()
        gvz_std = gvz_std.replace(0, np.nan)
        gvz_z = (gvz - gvz_mean) / gvz_std
        
        # GVZ 边际变化衰竭确认
        gvz_ma3 = gvz.rolling(window=3, min_periods=1).mean()
        
        # ----------------------------------------------------
        # 铁律2: 二阶导数捕捉衰竭 (绝不接飞刀)
        # ----------------------------------------------------
        
        # 做多条件 A: 衰退/流动性恐慌极值 + 开始衰竭
        # VIX 远高于 GVZ, 随后差值开始回落 (流动性冲击消退)
        liq_panic_extreme = spread_z > 1.7
        liq_panic_exhausted = spread < spread_ma3
        
        # 做多条件 B: 通胀/实际利率恐慌极值 + 开始衰竭
        # GVZ 远高于 VIX (差值极度负偏), 随后差值反弹回升 (紧缩恐慌消退)
        rate_panic_extreme = spread_z < -1.7
        rate_panic_exhausted = spread > spread_ma3
        
        # 做空条件 C: 通胀/利率极度自满 + 自满打破
        # 黄金波动率极低(无人对冲通胀/利率风险), 随后开始抬头 (加息预期萌芽)
        inflation_complacency_extreme = gvz_z < -1.7
        inflation_complacency_broken = gvz > gvz_ma3
        
        # 综合逻辑
        buy_cond = (liq_panic_extreme & liq_panic_exhausted) | (rate_panic_extreme & rate_panic_exhausted)
        short_cond = inflation_complacency_extreme & inflation_complacency_broken
        
        # 赋值信号 (严格只在触发时赋值脉冲)
        signal[short_cond] = -1.0
        signal[buy_cond] = 1.0  # 若极端日冲突，极值恐慌衰竭的做多赔率更高，设为优先
        
        # 填充潜在的 NaN 并确保序列名称正确
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"