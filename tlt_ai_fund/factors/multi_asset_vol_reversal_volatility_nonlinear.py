import numpy as np
import pandas as pd

class MultiAssetVolReversalFactor:
    """多资产波动率恐慌衰竭因子 (volatility/nonlinear)

    逻辑: 监控跨资产波动率的极端狂飙。在股市波动率(VIX)达到年度极值(Z-Score > 2.5)且开始回落时，
          通过黄金波动率(GVZCLS)的同步回落来确认跨资产"全面恐慌"的瓦解。
          由于波动率飙升期美债往往遭遇无差别流动性抛售(如2020年3月或2022年6月)，因此绝对禁止"极值即买入"。
          必须等待极值且二阶导数翻负(恐慌动量衰竭)，此时做多美债(TLT)能完美捕捉避险资金重返的脉冲主升浪。
          常态下输出 0.0，仅在极端事件反转瞬间及随后 5 天产生 +1.0 的狙击手脉冲。
    数据: vixcls, gvzcls
    触发: VIX Z-Score > 2.5 且 VIX 回落(跨过3日均线) 且 GVZCLS 同步回落
    输出: +1.0 (脉冲型看多TLT), 其余时间 0.0
    """

    def __init__(self):
        self.name = 'multi_asset_vol_reversal_volatility_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 序列，严格遵守"零值休眠"铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 前向填充缺失值，避免交易日历错位导致计算断层
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 1. 计算长周期(252日/一年) Z-Score，使用 min_periods=63 保证预热期后即可运算
        vix_mean = vix.rolling(window=252, min_periods=63).mean()
        vix_std = vix.rolling(window=252, min_periods=63).std()
        vix_zscore = (vix - vix_mean) / (vix_std + 1e-8)
        
        # 2. 衰竭与边际变化 (反接飞刀铁律：二阶导数翻负 + 脱离高水位均线)
        # 条件：不仅要单日环比下降(diff < 0)，还必须跌破短期(3日)均线，过滤极值区的"日内震荡假摔"
        vix_exhaustion = (vix.diff() < 0) & (vix < vix.rolling(window=3).mean())
        gvz_exhaustion = (gvz.diff() < 0) & (gvz < gvz.rolling(window=3).mean())
        
        # 3. 极值状态判定 (定义真正的极端恐慌)
        vix_extreme = vix_zscore > 2.5
        
        # 4. 非线性交叉确认：VIX极端高位 + VIX动量衰竭 + 黄金波动率同步衰竭
        # (跨域验证过滤掉单纯的股市下跌，必须是宏观跨资产的流动性冲击缓解)
        trigger = vix_extreme & vix_exhaustion & gvz_exhaustion
        
        # 5. 狙击手脉冲延展 (目标 5%~15% Trigger Rate)
        # 恐慌瓦解后的资金回流避险资产通常持续一周左右，向后延展 5 个交易日的脉冲信号
        extended_trigger = trigger.rolling(window=5, min_periods=1).max().fillna(0).astype(bool)
        
        # 触发日及顺延期赋值 +1.0 (看多美债)
        signal.loc[extended_trigger] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"