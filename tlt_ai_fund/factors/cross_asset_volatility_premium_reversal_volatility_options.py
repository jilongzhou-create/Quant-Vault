import numpy as np
import pandas as pd

class CrossAssetVolatilityPremiumReversalFactor:
    """跨资产波动率溢价反转因子 (volatility/options)

    逻辑: VIX(标普隐含波动率)与GVZCLS(黄金隐含波动率)的差值代表了风险资产相对避险实物资产的流动性恐慌溢价。当溢价极高且开始衰竭时，标志着对冲盘解体的流动性休克见顶，资金流回核心避险资产（看多美债）；当溢价极低且开始抬头时，标志着极端自满/投机情绪破裂，往往伴随紧缩周期的重估或滞胀定价（看空美债）。设计为狙击手脉冲，避免常态持仓。
    数据: vixcls, gvzcls
    触发: 多头: 溢价252日 Z-Score > 2.0 且 差值 < 3日均值 且 diff() < 0。空头: Z-Score < -2.0 且 差值 > 3日均值 且 diff() > 0。
    输出: +1.0 (恐慌衰竭, 买入TLT), -1.0 (自满破裂, 卖出TLT), 0.0 (常态休眠)
    """

    def __init__(self):
        self.name = 'cross_asset_vol_premium_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须处理缺少所需列的情况
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 数据预处理
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 构建跨资产波动率溢价
        vol_premium = vix - gvz
        
        # 避免边际初始数据不足, min_periods 控制平滑启动
        roll_mean = vol_premium.rolling(window=252, min_periods=60).mean()
        roll_std = vol_premium.rolling(window=252, min_periods=60).std()
        roll_std = roll_std.replace(0.0, np.nan)  # 避免除以0
        
        # 长周期水位极值评估
        z_score = (vol_premium - roll_mean) / roll_std
        
        # 短期边际动量衰竭/爆发评估 (二阶导数铁律)
        ma_3 = vol_premium.rolling(window=3, min_periods=1).mean()
        diff_1 = vol_premium.diff()
        
        # 多头条件: 跨资产恐慌溢价处于极端高位 (Z > 2.0) 且 动量开始衰竭回落
        long_cond = (z_score > 2.0) & (vol_premium < ma_3) & (diff_1 < 0)
        
        # 空头条件: 跨资产恐慌溢价处于极端低位 (黄金波动率异常强势, Z < -2.0) 且 自满被打破开始反转
        short_cond = (z_score < -2.0) & (vol_premium > ma_3) & (diff_1 > 0)
        
        # 赋值脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"