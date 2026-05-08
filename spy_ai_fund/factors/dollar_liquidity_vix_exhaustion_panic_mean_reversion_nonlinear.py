import numpy as np
import pandas as pd

class CrossAssetVolDivergenceFactor:
    """跨资产波动率背离衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合美股抄底与阴跌的非对称物理属性。当VIX极高且黄金波动率(GVZ)也偏高时，代表发生系统性流动性冲击(无差别抛售), 此时VIX显著见顶回落构成SPY极佳的极值衰竭抄底买点; 
          当VIX处于中等水位且单日跳升(轻微恐慌恶化)，但GVZ并未跟随上升时，代表仅是权益市场单边缩量阴跌，缺乏恐慌盘彻底出清，此时触发做空脉冲。
    数据: vixcls, gvzcls
    输出: +1.0 (系统性恐慌见顶衰竭, 强看多), -1.0 (权益单边阴跌发酵, 强看空), 0.0 (常态休眠)
    触发条件: 依赖单日VIX跳变(>1.0或<-1.0)保证脉冲属性，预估Trigger Rate 5%-15%
    """

    def __init__(self):
        self.name = 'cross_asset_vol_divergence_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查必需字段是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        # 填充缺失值，避免序列中断
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算252日滚动Z-Score (衡量一年的历史相对水位，具有明确经济学意义)
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)
        
        gvz_mean = gvz.rolling(window=252, min_periods=60).mean()
        gvz_std = gvz.rolling(window=252, min_periods=60).std()
        gvz_z = (gvz - gvz_mean) / (gvz_std + 1e-6)
        
        # 边际变化与动量 (捕捉事件瞬间)
        vix_diff1 = vix.diff(1)
        vix_ma3 = vix.rolling(window=3, min_periods=2).mean()
        gvz_diff3 = gvz.diff(3)
        
        signal = pd.Series(0.0, index=data.index)
        
        # 多头脉冲 (+1.0): 系统性恐慌的极值 + 衰竭
        # 1. 股市处于极端恐慌(Z>1.5) 且 黄金也受流动性冲击波及(Z>0.5)
        # 2. VIX今日显著回落(<-1.0点)，且低于3日均线，确认杀跌动量耗尽，防接飞刀
        is_systemic_panic = (vix_z > 1.5) & (gvz_z > 0.5)
        is_exhausted = (vix_diff1 < -1.0) & (vix < vix_ma3)
        buy_cond = is_systemic_panic & is_exhausted
        
        # 空头脉冲 (-1.0): 仅限权益资产的温水煮青蛙 (缓慢出血)
        # 1. VIX处于轻度至中度恐慌区间 (0.5 到 1.8)
        # 2. VIX今日显著跳升(>1.0点)且突破3日均线，恐慌正在稳步发酵
        # 3. 黄金波动率并未飙升(Z<=0.5)或近期回落，证明没有引发跨市场恐慌出清，仅是SPY单边流血
        is_mild_equity_fear = (vix_z > 0.5) & (vix_z <= 1.8)
        is_equity_worsening = (vix_diff1 > 1.0) & (vix > vix_ma3)
        is_no_systemic_fear = (gvz_z <= 0.5) | (gvz_diff3 < 0.0)
        sell_cond = is_mild_equity_fear & is_equity_worsening & is_no_systemic_fear
        
        # 信号赋值，默认0.0休眠
        signal[buy_cond] = 1.0
        signal[sell_cond & ~buy_cond] = -1.0
        
        # 确保无前瞻偏差，处理潜在NaN
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"