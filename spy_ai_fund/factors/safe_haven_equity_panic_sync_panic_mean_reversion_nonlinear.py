import numpy as np
import pandas as pd

class SafeHavenEquityPanicSyncFactor:
    """Safe Haven and Equity Panic Sync Factor (panic_mean_reversion/nonlinear)

    逻辑: 股票(VIX)与黄金避险(GVZ)的波动率同步飙升, 往往表明系统性流动性挤兑或极端宏观恐慌的到来。此时盲目抄底如同接飞刀。只有当两者的恐慌极值同步见顶并出现边际衰竭(低于短期均值)时, 才能确认系统性警报解除, 输出强看多脉冲; 若二者同处中高位且持续攀升创新高, 说明流动性危机正在共振发酵, 输出看空脉冲。
    数据: vixcls, gvzcls
    输出: +1.0(极端恐慌极值+同步衰竭, 抄底买点), -1.0(恐慌共振恶化, 避险抛售点), 0.0(常态休眠)
    触发条件: 波动率双双处于滚动252日的高Z-Score且当日边际回落触发买入脉冲, 处于中高位且同时破3日新高触发卖出脉冲。预期 Trigger Rate 控制在 5%-15%。
    """

    def __init__(self):
        self.name = 'safe_haven_equity_panic_sync'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 零值休眠铁律: 常态默认为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 避免前向偏差, 使用闭区间历史滚动计算状态 (Z-Score)
        vix_ma252 = vix.rolling(window=252, min_periods=126).mean()
        vix_std252 = vix.rolling(window=252, min_periods=126).std()
        vix_z = (vix - vix_ma252) / vix_std252.replace(0, np.nan)
        
        gvz_ma252 = gvz.rolling(window=252, min_periods=126).mean()
        gvz_std252 = gvz.rolling(window=252, min_periods=126).std()
        gvz_z = (gvz - gvz_ma252) / gvz_std252.replace(0, np.nan)
        
        # 二阶导数衰竭条件 (防接飞刀: 当天值显著低于近期均线, 说明脉冲冲顶结束)
        vix_3d_ma = vix.rolling(window=3, min_periods=2).mean()
        gvz_3d_ma = gvz.rolling(window=3, min_periods=2).mean()
        vix_exhausted = vix < vix_3d_ma
        gvz_exhausted = gvz < gvz_3d_ma
        
        # 恶化发酵条件 (钝刀割肉/主跌浪发酵: 突破短期新高)
        vix_worsening = vix > vix.rolling(window=3, min_periods=2).max().shift(1)
        gvz_worsening = gvz > gvz.rolling(window=3, min_periods=2).max().shift(1)
        
        # 脉冲买点: 双波幅极值(Z>1.2 属极端危机状态) + 同步回落衰竭
        buy_cond = (vix_z > 1.2) & (gvz_z > 1.2) & vix_exhausted & gvz_exhausted
        
        # 脉冲卖点: 双波幅偏高位(中轻度恐慌 Z>0.5) + 同步共振攀升恶化
        sell_cond = (vix_z > 0.5) & (gvz_z > 0.5) & vix_worsening & gvz_worsening
        
        # 信号赋值
        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0
        
        # 处理异常值，确保信号纯净
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"