import numpy as np
import pandas as pd

class VolatilityCrowdingReversalFactor:
    """波动率极值与拥挤反转 (volatility/options)

    逻辑: 监控跨资产(美股与黄金)期权隐含波动率的极端狂飙与死寂。常态下零值休眠。
          当波动率极端飙升时(恐慌挤兑), 美债可能因流动性冲击被抛售; 必须等待波动率同步回落的衰竭瞬间, 
          确认流动性危机瓦解, 避险资金重新安稳回流美债, 触发 +1.0 看多脉冲。
          反之, 波动率长期死寂后的同步爆发, 预示着Risk Parity去杠杆及风险偏好反转, 触发 -1.0 看空脉冲。
    数据: vixcls (美股期权隐含波动率), gvzcls (黄金期权隐含波动率)
    触发: 多头脉冲 -> 126日 Z-Score > 2.5 且双双回落至3日均线以下且日变为负 (衰竭确认)。
          空头脉冲 -> 126日 Z-Score < -1.5 且双双突破5日均线且日变为正 (爆发确认)。
    输出: 严格脉冲型信号, 仅极端反转瞬间输出 +1.0 或 -1.0, 其余时间常态 0.0。
    """

    def __init__(self):
        self.name = 'volatility_crowding_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须同时拥有美股和黄金期权波动率数据才能做跨资产确认
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 计算 126 日 (半年) 滚动 Z-Score 以捕捉中期波动率水位的极端拥挤
        vix_z = (vix - vix.rolling(126).mean()) / vix.rolling(126).std()
        gvz_z = (gvz - gvz.rolling(126).mean()) / gvz.rolling(126).std()

        # 计算短期均线, 用于二阶导数(Anti-Catch-Falling-Knife)判断
        vix_ma3 = vix.rolling(3).mean()
        gvz_ma3 = gvz.rolling(3).mean()
        
        vix_ma5 = vix.rolling(5).mean()
        gvz_ma5 = gvz.rolling(5).mean()

        # 条件1: 多头脉冲 (极值 + 衰竭)
        # 绝对禁止接飞刀: 必须在 Z-Score > 2.5 且动量开始往下走(diff < 0 且跌破均线)才触发
        long_extreme = (vix_z > 2.5) | (gvz_z > 2.5)
        long_exhaustion = (vix < vix_ma3) & (gvz < gvz_ma3) & (vix.diff() < 0) & (gvz.diff() < 0)
        long_cond = long_extreme & long_exhaustion

        # 条件2: 空头脉冲 (死寂 + 爆发)
        # 波动率极低(拥挤做空波动率)且开始抬头, 捕捉变盘瞬间的边际变化
        short_extreme = (vix_z < -1.5) & (gvz_z < -1.5)
        short_breakout = (vix > vix_ma5) & (gvz > gvz_ma5) & (vix.diff() > 0) & (gvz.diff() > 0)
        short_cond = short_extreme & short_breakout

        # 零值休眠铁律: 默认全为0, 只有瞬间触发+1.0或-1.0
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"