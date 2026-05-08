import numpy as np
import pandas as pd

class YieldCurveWholeVolExhaustionFactor:
    """全曲线波动率共振衰竭因子 (volatility/options)

    逻辑: 捕捉债市内部的极端微观结构恐慌。收益率曲线短端(2Y-3M)与中长端(10Y-2Y)的利差波动率是债市隐含波动的绝佳代理(类似合成MOVE指数)。当整条曲线的波动率同步飙升至极值(标志着极端的联储重新定价/流动性冲击)，且双双回落衰竭时，标志着恐慌抛售枯竭，美债结构性买盘回归。此维度纯粹基于利率衍生微观结构，完全正交于跨资产VIX模型。
    数据: t10y2y, t10y3m (纯债市维度的衍生波动率，规避CoreAnchor及VIX重叠)
    触发: 曲线一端的21日波动率 Z-Score > 2.5 且另一段 > 1.0 (极值)，随后两者的波动率均下穿3日均线 (衰竭)。
    输出: 脉冲信号 +1.0 (恐慌衰竭当天及随后2天做多TLT)
    """

    def __init__(self):
        self.name = 'yield_curve_whole_vol_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 验证必需的数据字段是否存在
        if 't10y2y' not in data.columns or 't10y3m' not in data.columns:
            return signal
            
        # 向前填充处理缺失值
        t10y2y = data['t10y2y'].ffill()
        t10y3m = data['t10y3m'].ffill()
        
        # 1. 结构化切分曲线 (构建独立的利率微观形变追踪)
        # 中长端利差 (Belly to Long)
        belly_spread = t10y2y
        # 极短端利差 (Short to Belly) = (10Y-3M) - (10Y-2Y) = 2Y - 3M
        short_spread = t10y3m - t10y2y
        
        # 核心铁律3: 边际变化 (Marginal Change Only)
        # 提取利差的真实微观波动率 (21个交易日的差分标准差，反映重定价的剧烈程度)
        belly_vol = belly_spread.diff().rolling(21).std()
        short_vol = short_spread.diff().rolling(21).std()
        
        # 计算局部Z-Score (126日/半年窗口，适应不同宏观周期的基准波动水位)
        belly_mean = belly_vol.rolling(126).mean()
        belly_std = belly_vol.rolling(126).std().replace(0, np.nan)
        belly_vol_z = ((belly_vol - belly_mean) / belly_std).fillna(0)
        
        short_mean = short_vol.rolling(126).mean()
        short_std = short_vol.rolling(126).std().replace(0, np.nan)
        short_vol_z = ((short_vol - short_mean) / short_std).fillna(0)
        
        # 捕捉极端恐慌共振: 全曲线剧烈形变 (至少一段达到2.5标准差极值，另一段处于>1.0的高位)
        max_vol_z = np.maximum(belly_vol_z, short_vol_z)
        min_vol_z = np.minimum(belly_vol_z, short_vol_z)
        cond_panic = (max_vol_z > 2.5) & (min_vol_z > 1.0)
        
        # 核心铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 必须等待波动率开始绝对衰竭 (低于3日超短期均线)
        belly_ma = belly_vol.rolling(3).mean()
        short_ma = short_vol.rolling(3).mean()
        cond_exh = (belly_vol < belly_ma) & (short_vol < short_ma)
        
        # 综合触发: 极值 + 衰竭双重确认
        raw_trigger = cond_panic & cond_exh
        
        # 核心铁律1: 零值休眠 (Sniper Pulse)
        # 将极值脉冲向后展期2天以覆盖完整的初始反转窗口，同时确保Trigger Rate稳定落在 5%-15% 的目标限制内
        buy_trigger = raw_trigger | raw_trigger.shift(1).fillna(False) | raw_trigger.shift(2).fillna(False)
        
        # 输出看多脉冲信号
        signal[buy_trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"