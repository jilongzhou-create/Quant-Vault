import numpy as np
import pandas as pd

class OptionsImpliedVolSpreadShockFactor:
    """期权隐含波动率跨资产利差冲击因子 (unstructured/options)

    逻辑: 捕捉期权市场定价的跨资产极值错位以寻找美债脉冲拐点。股市波动率(VIX)代表经济增长与权益流动性风险，黄金波动率(GVZ)代表法币信用与滞胀风险。当VIX相对GVZ极度飙升时(差值Z-Score>2.5)，说明市场定价为纯粹的权益崩盘恐慌，此时若差值开始回落(二阶衰竭)，说明无差别杀跌阶段结束，触发买入TLT的避险配置脉冲。反之，当该差值极度低迷(Z-Score<-2.5)且开始反弹时，说明市场处于极度自满的“金发女孩”末期，任何微小的紧缩或通胀预期抬头都会导致长端美债遭抛售，触发做空TLT脉冲。
    数据: vixcls, gvzcls
    触发: VIX-GVZ差值的 252日 Z-Score > 2.5 且开始回落 (看多)；Z-Score < -2.5 且开始反弹 (看空)。
    输出: +1.0 (恐慌衰竭做多美债), -1.0 (自满衰竭做空美债), 0.0 (常态休眠)
    """

    def __init__(self, lookback_window: int = 252):
        self.name = 'options_implied_vol_spread_shock'
        self.lookback_window = lookback_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0.0 的休眠信号 Series
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 铁律3：边际变化。利差本身即是衡量跨资产相对强度的边际风险定价指标
        vol_spread = vix - gvz
        
        # 计算极值条件 (Z-Score)
        spread_mean = vol_spread.rolling(window=self.lookback_window, min_periods=self.lookback_window // 2).mean()
        spread_std = vol_spread.rolling(window=self.lookback_window, min_periods=self.lookback_window // 2).std()
        
        spread_z = (vol_spread - spread_mean) / (spread_std + 1e-8)
        
        # 铁律2：二阶导数防飞刀（必须等极值+衰竭同时满足）
        # 权益恐慌衰竭确认：差值在极度高位开始回落，且跌破3日均线
        spread_falling = (vol_spread.diff(1) < 0) & (vol_spread < vol_spread.rolling(3).mean())
        
        # 自满状态衰竭确认：差值在极度低位开始反弹，且突破3日均线
        spread_rising = (vol_spread.diff(1) > 0) & (vol_spread > vol_spread.rolling(3).mean())
        
        # 铁律1：狙击手级别脉冲，严格组合极值与二阶拐点
        long_condition = ((spread_z > 2.5) & spread_falling).fillna(False)
        short_condition = ((spread_z < -2.5) & spread_rising).fillna(False)
        
        # 仅在触发日赋值
        signal.loc[long_condition] = 1.0
        signal.loc[short_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback_window={self.lookback_window})"