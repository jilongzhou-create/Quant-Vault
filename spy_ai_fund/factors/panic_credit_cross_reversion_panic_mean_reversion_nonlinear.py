import numpy as np
import pandas as pd

class PanicCreditCrossReversionFactor:
    """恐慌与信用双重极值均值回归因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合股市恐慌(VIX)与信用市场融资压力(高收益债OAS)。在美股长牛与均值回归属性中，当股市与信用市场同时遭遇局部极端恐慌(Z-Score>1.5)且今日出现衰竭(指标回落)时，是高胜率的'黄金坑'抄底买点；若压力脱离舒适区温和攀升(钝刀割肉)，则属于主跌浪发酵期，输出看空。
    数据: vixcls (CBOE VIX), bamlh0a0hym2 (US High Yield OAS)
    输出: 强看多(+1.0)代表极端恐慌极值衰竭，看空(-1.0)代表轻度恐慌趋势发酵，常态返回0.0。
    触发条件: 非线性交叉压力状态(Z-Score)与动量转折(日度Diff)，且经过脉冲去重处理，预期 Trigger Rate 严格控制在 5%-15% 之间。
    """

    def __init__(self, lookback_window: int = 63, extreme_z: float = 1.5, mild_z: float = 0.5):
        self.name = 'panic_credit_cross_reversion'
        # 63个交易日代表一个季度的局部宏观状态窗口
        self.lookback_window = lookback_window
        self.extreme_z = extreme_z
        self.mild_z = mild_z

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少核心维度的数据，则直接返回常态 0.0
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 提取并前向填充数据，对齐交易日历
        vix = data['vixcls'].ffill()
        oas = data['bamlh0a0hym2'].ffill()

        # 1. 计算局部 Z-Score，反映当期偏离常态的压力水平
        vix_mean = vix.rolling(window=self.lookback_window).mean()
        vix_std = vix.rolling(window=self.lookback_window).std().replace(0, 1e-5)
        vix_z = (vix - vix_mean) / vix_std

        oas_mean = oas.rolling(window=self.lookback_window).mean()
        oas_std = oas.rolling(window=self.lookback_window).std().replace(0, 1e-5)
        oas_z = (oas - oas_mean) / oas_std

        # 综合系统性压力指数 (股市流动性 + 实体融资信用)
        stress_level = (vix_z + oas_z) / 2.0

        # 2. 计算边际变化(二阶导数)，防接飞刀，捕捉动量衰竭
        vix_diff_1 = vix.diff(1)
        vix_diff_3 = vix.diff(3)
        oas_diff_1 = oas.diff(1)

        # 3. 核心触发逻辑
        
        # 买入条件: 极值 + 衰竭
        # 压力指数极高 (> 1.5 Std)，且今日 VIX 回落，同时 OAS 停止走阔
        buy_cond = (
            (stress_level > self.extreme_z) & 
            (vix_diff_1 < 0) & 
            (oas_diff_1 <= 0)
        )

        # 卖出条件: 轻度恐慌发酵
        # 压力指数温和脱离均值 (0.5 ~ 1.5 Std)，且 VIX 近3天明确上行，OAS 今日走阔 (主跌浪前兆)
        sell_cond = (
            (stress_level > self.mild_z) & 
            (stress_level <= self.extreme_z) & 
            (vix_diff_3 > 0) & 
            (oas_diff_1 > 0)
        )

        # 4. 初始化信号并实施严格的零值脉冲约束
        signal = pd.Series(0.0, index=data.index)
        
        # 脉冲触发过滤器：只在状态改变的第一天(瞬间)释放能量
        buy_pulse = buy_cond & ~buy_cond.shift(1).fillna(False)
        sell_pulse = sell_cond & ~sell_cond.shift(1).fillna(False)

        signal[buy_pulse] = 1.0
        signal[sell_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.lookback_window}, extreme_z={self.extreme_z}, mild_z={self.mild_z})"