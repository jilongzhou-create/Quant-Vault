import numpy as np
import pandas as pd

class OptionsImpliedVolSpreadReversalFactor:
    """期权隐含波动率跨资产利差反转因子 (volatility/options)

    逻辑: 监控期权隐含的跨资产恐慌差值 (VIX-GVZ)。当VIX相对GVZ极度狂飙时代表通缩型流动性冲击，其峰值衰竭且开始回落的瞬间，往往意味着恐慌性抛售结束、央行预期转鸽，此时爆发美债(TLT)的强烈做多脉冲；反之GVZ相对狂飙代表极端的滞胀型通胀恐慌，反转时利空美债。此设计纯粹捕捉预期改变瞬间，避免接飞刀。
    数据: vixcls (CBOE VIX 股票隐含波动率), gvzcls (CBOE 黄金隐含波动率)
    触发: VIX-GVZ差值的252日 Z-Score > 2.5 且 差值diff() < 0 且低于3日均值(衰竭) -> +1.0 (看多美债)；反之则 -> -1.0 (看空美债)
    输出: [-1.0, 1.0] 的极值衰竭脉冲信号 (非连续)
    """

    def __init__(self):
        self.name = 'options_implied_vol_spread_reversal'
        self.zscore_window = 252
        self.exhaustion_window = 3
        self.zscore_threshold = 2.5

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse) - 初始化全为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖数据是否存在
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        # 填充缺失值，对齐交易日历
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 计算期权隐含波动率利差 (股票恐慌 vs 避险资产恐慌)
        vol_spread = vix - gvz

        # 计算 252 日滚动 Z-Score (捕捉绝对极值)
        rolling_mean = vol_spread.rolling(window=self.zscore_window, min_periods=60).mean()
        rolling_std = vol_spread.rolling(window=self.zscore_window, min_periods=60).std()
        
        # 避免除以 0
        rolling_std = rolling_std.replace(0.0, np.nan)
        spread_zscore = (vol_spread - rolling_mean) / rolling_std

        # 铁律3: 边际变化 - 必须使用一阶差分计算动量
        spread_diff = vol_spread.diff()
        
        # 铁律2: 二阶导数 - 用于确认飞刀动能衰竭
        spread_ma = vol_spread.rolling(window=self.exhaustion_window).mean()

        # 看多美债 (TLT) 脉冲条件: 
        # 1. 相对恐慌极度高昂 (Z-Score > 2.5) 
        # 2. 恐慌动能已经停止并开始回落 (diff < 0) 
        # 3. 跌破短期均线确认 (vol_spread < 3日均值)
        long_pulse = (
            (spread_zscore > self.zscore_threshold) & 
            (spread_diff < 0) & 
            (vol_spread < spread_ma)
        )

        # 看空美债 (TLT) 脉冲条件: 
        # 1. 相对恐慌极度低迷 (Z-Score < -2.5) 
        # 2. 动能逆转向上 (diff > 0)
        # 3. 突破短期均线确认 (vol_spread > 3日均值)
        short_pulse = (
            (spread_zscore < -self.zscore_threshold) & 
            (spread_diff > 0) & 
            (vol_spread > spread_ma)
        )

        # 触发脉冲信号
        signal.loc[long_pulse] = 1.0
        signal.loc[short_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, threshold={self.zscore_threshold})"