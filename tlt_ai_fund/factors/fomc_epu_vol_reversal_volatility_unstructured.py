import numpy as np
import pandas as pd

class FomcEpuVolReversalFactor:
    """FomcEpuVolReversalFactor (volatility/unstructured)

    逻辑: 结合非结构化文本(EPU)与跨资产波动率(GVZCLS)捕捉恐慌极值。当宏观不确定性与避险波动率同步见顶衰竭时，市场风险溢价急速消退，资产将沿着近期FOMC情绪的边际变化方向发生剧烈重定价。因子通过严格捕捉不确定性瓦解的瞬间发射多空脉冲。
    数据: usepuindxd (经济政策不确定性指数), gvzcls (黄金波动率), fomc_sentiment (FOMC鹰鸽得分)
    触发: EPU的 252日 Z-Score > 1.5 (极值) + EPU与GVZCLS同时回落 (二阶导数衰竭) + FOMC情绪季度边际变化 (动量方向)。
    输出: 脉冲信号。宽松边际下的恐慌瓦解买入美债 (+1.0)，紧缩边际下的恐慌瓦解抛售美债 (-1.0)。
    """

    def __init__(self):
        self.name = 'fomc_epu_vol_reversal_factor'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 处理数据缺失，如果缺少任何必需的特征列，则返回全 0 的休眠信号
        required_cols = ['usepuindxd', 'gvzcls', 'fomc_sentiment']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0.0, index=data.index, name=self.name)

        # 避免未来数据，使用前向填充保证时序有效对齐
        df = data[required_cols].ffill()
        
        epu = df['usepuindxd']
        gvz = df['gvzcls']
        fomc = df['fomc_sentiment']

        # ---------------------------------------------------------
        # 铁律2条件A: 计算核心因子的极端状态 (极值狂飙)
        # ---------------------------------------------------------
        epu_roll_mean = epu.rolling(window=252, min_periods=63).mean()
        epu_roll_std = epu.rolling(window=252, min_periods=63).std()
        
        # 避免常数区间导致的标准差为0的除零错误
        epu_roll_std = epu_roll_std.replace(0, np.nan)
        epu_zscore = (epu - epu_roll_mean) / epu_roll_std

        # EPU 达到 1.5 个标准差以上，锁定由于极端事件引发的不确定性飙升区域
        is_epu_extreme = epu_zscore > 1.5

        # ---------------------------------------------------------
        # 铁律2条件B: 二阶导数衰竭 (严禁接飞刀，必须等动能回落)
        # ---------------------------------------------------------
        # 本域衰竭：政策不确定性指数开始从高点回落且跌破近3日均线
        is_epu_exhausted = (epu < epu.rolling(window=3).mean()) & (epu.diff(1) < 0)
        
        # 跨域确认：避险资产(黄金)的波动率在近3日内发生同步实质性回落
        is_gvz_exhausted = gvz.diff(3) < 0

        # ---------------------------------------------------------
        # 铁律3: 边际变化判定基调方向 (绝对禁止输出/判断水平值)
        # ---------------------------------------------------------
        # 使用最近一个季度(63个交易日) FOMC情绪得分的边际变化总量。
        # 此动量代表大波段上的"预期转鸽"或"预期转鹰"。
        fomc_momentum = fomc.diff(63)
        
        # ---------------------------------------------------------
        # 信号合成与铁律1: 零值休眠 (Sniper Pulse)
        # ---------------------------------------------------------
        signal = pd.Series(0.0, index=data.index)
        
        # 触发脉冲看多 (避险退潮 + 宽松预期确立 = 长债大反攻)
        bull_trigger = is_epu_extreme & is_epu_exhausted & is_gvz_exhausted & (fomc_momentum > 0.05)
        
        # 触发脉冲看空 (避险退潮 + 紧缩预期确立 = 重拾加息定价抛售长债)
        bear_trigger = is_epu_extreme & is_epu_exhausted & is_gvz_exhausted & (fomc_momentum < -0.05)

        signal[bull_trigger] = 1.0
        signal[bear_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"