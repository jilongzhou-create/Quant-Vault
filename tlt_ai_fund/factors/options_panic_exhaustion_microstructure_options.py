import numpy as np
import pandas as pd

class OptionsPanicExhaustionFactor:
    """期权恐慌衰竭因子 (microstructure/options)

    逻辑: 捕捉期权市场(美股VIX+黄金GVZ)系统性恐慌极值后的衰竭瞬间。当跨资产隐含波动率飙升至极端水平并开始回落时, 标志着流动性冲击(Dash for Cash)的结束, 避险资金将重新有序流入美债(TLT), 此时产生看多脉冲。
    数据: vixcls (标普500隐含波动率), gvzcls (黄金ETF隐含波动率)
    触发: (VIX+GVZ) 的 252日 Z-Score > 2.0 且 当日值 < 3日均值 且 边际变化(diff) < 0
    输出: 脉冲型信号, 恐慌衰竭见顶回落瞬间输出 +1.0, 常态严格休眠为 0.0
    """

    def __init__(self, z_threshold: float = 2.0, window: int = 252, smooth_window: int = 3):
        self.name = 'options_panic_exhaustion'
        self.z_threshold = z_threshold
        self.window = window
        self.smooth_window = smooth_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 初始信号严格为 0.0, 只有触发日才会有非零输出
        signal = pd.Series(0.0, index=data.index)

        # 数据完整性检查
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal

        # 提取微观结构数据并填充缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()

        # 构建跨资产期权市场总压力指数 (Systemic Options Stress)
        # 股票(风险)与黄金(避险)期权波动率同时飙升代表严重的系统性流动性危机
        agg_stress = vix + gvz

        # 计算 252 日 (一年) 滚动 Z-Score 确定绝对水位极值
        roll_mean = agg_stress.rolling(window=self.window).mean()
        roll_std = agg_stress.rolling(window=self.window).std()
        
        # 避免除以零产生的无穷大
        roll_std = roll_std.replace(0.0, np.nan)
        z_score = (agg_stress - roll_mean) / roll_std

        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife) & 铁律3: 边际变化
        # 绝对禁止在波动率飙升途中接飞刀, 必须等待其跌破短期均线且边际动量转负
        stress_ma = agg_stress.rolling(window=self.smooth_window).mean()
        diff = agg_stress.diff()

        is_exhausting = (agg_stress < stress_ma) & (diff < 0)

        # 脉冲触发: 极值条件 与 衰竭条件 同时满足
        long_condition = (z_score > self.z_threshold) & is_exhausting

        # 触发看多美债脉冲
        signal[long_condition] = 1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_threshold={self.z_threshold}, window={self.window}, smooth_window={self.smooth_window})"