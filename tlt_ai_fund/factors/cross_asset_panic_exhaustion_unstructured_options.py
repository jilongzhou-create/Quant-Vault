import numpy as np
import pandas as pd

class MacroFearExhaustionOptionsFactor:
    """宏观恐慌衰竭因子 (unstructured/options)

    逻辑: 结合非结构化新闻(EPU经济政策不确定性)与黄金期权隐含波动率(GVZ)构建宏观滞胀恐慌指数。当极度恐慌(由于政策或通胀失控)达到极值并首次向下交叉3日均线(恐慌脉冲衰竭)时, 且2年期美债收益率确认开启下行, 则生成一次看多美债的狙击脉冲。反之做空。完全规避了常规VIX的微观结构, 引入全新的基本面宏观维度。
    数据: usepuindxd (非结构化新闻), gvzcls (期权衍生), dgs2 (预期确认)
    触发: (EPU_z + GVZ_z) 5日最高 > 1.5, 且当日首次下穿3日均线, 且 dgs2.diff(3) < 0
    输出: [-1.0, 1.0] 的严格稀疏脉冲信号
    """

    def __init__(self):
        self.name = 'macro_fear_exhaustion_unstructured_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠初始状态
        signal = pd.Series(0.0, index=data.index)

        # 校验所需数据列
        required_cols = ['usepuindxd', 'gvzcls', 'dgs2']
        for col in required_cols:
            if col not in data.columns:
                return signal
        
        # 填充缺失值以防止跨周末/节假日导致的断层
        epu = data['usepuindxd'].ffill()
        gvz = data['gvzcls'].ffill()
        dgs2 = data['dgs2'].ffill()

        # 1. 快速滚动 Z-Score (捕捉边际极端突变)
        window = 42
        epu_z = (epu - epu.rolling(window).mean()) / epu.rolling(window).std().replace(0, np.nan)
        gvz_z = (gvz - gvz.rolling(window).mean()) / gvz.rolling(window).std().replace(0, np.nan)

        # 处理可能因早期数据缺失导致的 NaN (特别是 GVZ 在 2008 年前为空)
        epu_z = epu_z.fillna(0.0)
        gvz_z = gvz_z.fillna(0.0)

        # 混合滞胀/宏观政策恐慌指数 (跨域融合, 完全正交于股票VIX)
        fear_index = epu_z + gvz_z

        # 2. 极端区域判断 (铁律2: Anti-Catch-Falling-Knife 第一步 - 必须处于极值区)
        # 设定为 1.5 阈值以保证满足 5% - 15% 的目标 Trigger Rate
        long_extreme = fear_index.rolling(5).max() > 1.5
        short_extreme = fear_index.rolling(5).min() < -1.5

        # 3. 脉冲衰竭条件 (铁律1: Sniper Pulse 零值休眠核心逻辑 - 仅在穿越均线当天触发脉冲)
        fear_ma = fear_index.rolling(3).mean()
        
        # 下穿: 昨天还在均线之上/平齐, 今天跌破均线 -> 宏观恐慌正式步入衰竭
        cross_under = (fear_index < fear_ma) & (fear_index.shift(1) >= fear_ma.shift(1))
        
        # 上穿: 昨天还在均线之下/平齐, 今天升破均线 -> 极度自满正式结束
        cross_over = (fear_index > fear_ma) & (fear_index.shift(1) <= fear_ma.shift(1))

        # 4. 预期边际变化交叉验证 (铁律3: 只有预期发生改变瞬间才触发信号)
        # dgs2 短端利率对政策预期最敏感
        dgs2_mom = dgs2.diff(3)
        long_confirm = dgs2_mom < 0.0  # 确认短端利率下行 (降息预期骤升)
        short_confirm = dgs2_mom > 0.0 # 确认短端利率上行 (紧缩预期骤升)

        # 5. 组合信号生成 (同时满足极端值、拐点衰竭与跨资产验证)
        long_pulse = long_extreme & cross_under & long_confirm
        short_pulse = short_extreme & cross_over & short_confirm

        # 严格赋予脉冲信号
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"