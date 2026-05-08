import numpy as np
import pandas as pd

class VolatilityCurveReversalFactor:
    """Volatility and Yield Curve Microstructure Reversal

    逻辑: 结合跨资产波动率极值与收益率曲线边际变化，动态捕捉债市的极端拐点。在增长/流动性恐慌时（VIX与黄金波动率同步飙升后回落，且收益率曲线边际陡峭化定价降息），触发看多美债脉冲；在通胀/紧缩冲击时（波动率抬头且收益率曲线剧烈平坦化），触发看空美债脉冲。这解决了单纯看VIX无法区分“避险模式”与“加息通胀模式”的致命缺陷。信号生成后维持极短几日。
    数据: vixcls (VIX), gvzcls (黄金隐含波动率), t10y2y (期限利差)
    触发: 多头=VIX_Z>1.5+波动率衰竭+曲线变陡(>2bps); 空头=曲线剧烈平坦化(<-4bps)+波动率抬头/非衰竭。
    输出: 狙击型脉冲信号 [-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'volatility_curve_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 验证依赖数据
        req_cols = ['vixcls', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)
        
        vix = data['vixcls'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 如果存在黄金波动率则使用，否则用VIX自身占位（保证逻辑能够平稳降级通过）
        if 'gvzcls' in data.columns:
            gvz = data['gvzcls'].ffill()
        else:
            gvz = vix
            
        # 2. 计算极值指标 (126日约为半年窗口，捕捉中短期宏观周期的极端状态)
        vix_mean = vix.rolling(126).mean()
        vix_std = vix.rolling(126).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, 1e-5)
        
        gvz_mean = gvz.rolling(126).mean()
        gvz_std = gvz.rolling(126).std()
        gvz_z = (gvz - gvz_mean) / gvz_std.replace(0, 1e-5)
        
        # 3. 核心铁律：二阶导数与边际变化 (衰竭与发散)
        # 波动率回落表示恐慌/拥挤盘衰竭 (Anti-Catch-Falling-Knife)
        vix_exhaustion = vix.diff(1) < 0
        gvz_exhaustion = gvz.diff(1) < 0
        # 波动率抬头表示冲击开始
        vix_waking = vix.diff(1) > 0
        
        # 收益率曲线3日边际动量：>0为陡峭化(定价宽松/经济下行)，<0为平坦化(定价紧缩/加息预期)
        curve_change = t10y2y.diff(3)
        
        # 4. 触发逻辑
        # 看多美债：跨资产恐慌极值 + 恐慌开始衰竭 + 收益率曲线开始陡峭化(避险与宽松预期同时兑现)
        long_cond = (vix_z > 1.5) & (gvz_z > 1.0) & vix_exhaustion & gvz_exhaustion & (curve_change > 0.02)
        
        # 看空美债：分为自满破裂(紧缩突袭)与通胀冲击持续(如2022年股债双杀模式)
        short_complacency = (vix_z < -1.0) & vix_waking & (curve_change < -0.04)
        short_inflation_shock = (vix_z > 0.5) & (~vix_exhaustion) & (curve_change < -0.06)
        short_cond = short_complacency | short_inflation_shock
        
        # 5. 信号赋值 (初始为0.0，零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        # 6. 狙击手脉冲延展 (使得Trigger Rate落在 5% - 15% 目标区间)
        # 极端事件发生的当天及随后极短几天内(limit=2即持续3天)保持信号
        signal = signal.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"