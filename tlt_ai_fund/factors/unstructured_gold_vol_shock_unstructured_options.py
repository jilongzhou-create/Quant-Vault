import numpy as np
import pandas as pd

class UnstructuredGoldVolShockFactor:
    """Unstructured Gold Volatility Options Shock Factor (unstructured/options)

    逻辑: 黄金隐含波动率(GVZCLS)是捕捉非结构化宏观恐慌(如地缘冲突、黑天鹅极度避险)的期权衍生指标。
          非结构化冲击具有极强的爆发性，因此必须采用狙击手级的脉冲设计。当期权市场对黄金的避险买盘动能
          (5日边际变化量)出现极端脉冲时，意味着非结构化冲击达到顶峰。依据“二阶导数”铁律，只有当极端恐慌
          动能开始衰竭(GVZ绝对水位跌破3日均线)时，才确认流动性冲击的极点已过。此时市场将激进Price-in
          美联储的流动性注入与政策转向，驱动美债(TLT)迎来强劲的做多脉冲。反之，极端自满情绪衰竭时看空。
    数据: gvzcls (CBOE黄金ETF隐含波动率指数)
    触发: GVZ的5日变化量(1周动量)在63日(1个季度)窗口下 Z-Score > 2.5 且 当日GVZ < 3日均值(恐慌衰竭) -> +1.0
          GVZ的5日变化量在63日窗口下 Z-Score < -2.5 且 当日GVZ > 3日均值(自满反转) -> -1.0
    输出: [-1.0, 1.0] 的极值脉冲，非触发日常态严格休眠返回 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_gold_vol_shock_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化信号，严格遵守零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        if 'gvzcls' not in data.columns:
            return signal

        # 数据前向填充，防止因节假日造成的缺失
        gvz = data['gvzcls'].ffill()

        # 铁律3: 边际变化 (Marginal Change Only)
        # 使用 5个交易日 (1周) 的动量变化来捕捉期权市场恐慌情绪的突变瞬间
        momentum = gvz.diff(5)

        # 铁律1: 零值休眠 (Sniper Pulse)
        # 使用 63个交易日 (约1个季度) 滚动窗口计算局部宏观状态下的 Z-Score
        # 相比于252日，63日窗口能在肥尾分布下更好捕捉局部的突发极端事件，从而确保 Trigger Rate 在 5%-15% 之间
        roll_mean = momentum.rolling(window=63, min_periods=21).mean()
        roll_std = momentum.rolling(window=63, min_periods=21).std()

        # 计算 Z-Score，处理除零异常
        zscore = (momentum - roll_mean) / roll_std.replace(0, np.nan)

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 使用 3个交易日 移动平均线作为极值衰竭的判定基准
        gvz_ma3 = gvz.rolling(window=3).mean()
        
        # 衰竭条件判定
        is_panic_exhausting = gvz < gvz_ma3       # 恐慌开始回落
        is_complacency_exhausting = gvz > gvz_ma3 # 自满开始反弹

        # 生成脉冲触发条件
        # 看多条件：极端避险恐慌突发 (Z-Score > 2.5) 且 恐慌情绪出现衰竭拐点
        long_cond = (zscore > 2.5) & is_panic_exhausting
        
        # 看空条件：极端自满情绪突发 (Z-Score < -2.5) 且 自满情绪出现反转拐点
        short_cond = (zscore < -2.5) & is_complacency_exhausting

        # 赋值信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"