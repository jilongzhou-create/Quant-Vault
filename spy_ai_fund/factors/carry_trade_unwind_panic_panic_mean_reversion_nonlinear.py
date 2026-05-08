import numpy as np
import pandas as pd

class CarryTradeUnwindPanicFactor:
    """日元套息解盘恐慌衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合美元兑日元汇率(套息交易避险风向标)和VIX。当日元处于中期升值趋势(套息解盘、全球去杠杆)且VIX飙升至高位时，代表系统性流动性冲击。若此时出现二阶导衰竭(今日VIX回落且日元升值暂缓)，说明抛压枯竭，触发抄底；若VIX和日元仍在加速恶化，说明接飞刀风险极高，触发看空。
    数据: [vixcls, dexjpus]
    输出: [+1.0 恐慌衰竭抄底, -1.0 恐慌发酵趋势恶化, 0.0 无操作]
    触发条件: 股市恐慌极值 + 日元避险动能反转触发多头；中等波动 + 动量恶化触发空头。预期 Trigger Rate 在 8% - 12% 之间。
    """

    def __init__(self):
        self.name = 'carry_trade_unwind_panic_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        required_cols = ['vixcls', 'dexjpus']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        # 前向填充缺失值以处理不同交易日的对齐问题
        df = data[required_cols].ffill()
        
        vix = df['vixcls']
        jpy = df['dexjpus']

        if vix.isna().all() or jpy.isna().all():
            return pd.Series(0.0, index=data.index, name=self.name)

        # 1. 计算 VIX 的 252 日 (约1个交易年) Z-Score, 用于定位恐慌极端程度
        vix_mean_252 = vix.rolling(window=252, min_periods=60).mean()
        vix_std_252 = vix.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        vix_z = (vix - vix_mean_252) / vix_std_252

        # 2. 计算日元的中期趋势 (60交易日约一季度), 小于均线代表日元升值(美元贬值), 是避险去杠杆的宏观环境
        jpy_ma60 = jpy.rolling(window=60, min_periods=15).mean()

        # 3. 边际动量变化 (二阶导数铁律)
        vix_diff = vix.diff()
        jpy_diff = jpy.diff()

        # 构建多头条件: 极端恐慌 + 动能衰竭 (狙击手模式抄底)
        # - vix_z > 1.0: 股市处于 1 个标准差以上的极度恐慌高位
        # - vix_diff < 0: 今日 VIX 回落, 恐慌情绪见顶衰竭
        # - jpy < jpy_ma60: 宏观环境处于日元避险周期
        # - jpy_diff > 0: 今日日元贬值(DEXJPUS上升), 避险资金平仓枯竭
        c1_long = vix_z > 1.0
        c2_long = vix_diff < 0
        c3_long = jpy < jpy_ma60
        c4_long = jpy_diff > 0
        buy_condition = c1_long & c2_long & c3_long & c4_long

        # 构建空头条件: 恐慌酝酿 + 避险情绪持续恶化 (防接飞刀)
        # - vix_z 处于 (-0.8, 1.0]: 非极端恐慌期, 处于温和震荡或缓慢发酵阶段
        # - vix_diff > 0: VIX 今日仍在上升
        # - jpy < jpy_ma60: 处于日元避险周期
        # - jpy_diff < 0: 日元今日继续加速升值, 说明机构仍在不计成本去杠杆
        c1_short = (vix_z > -0.8) & (vix_z <= 1.0)
        c2_short = vix_diff > 0
        c3_short = jpy < jpy_ma60
        c4_short = jpy_diff < 0
        sell_condition = c1_short & c2_short & c3_short & c4_short

        # 组合脉冲信号
        signal = pd.Series(0.0, index=df.index, name=self.name)
        signal.loc[buy_condition] = 1.0
        signal.loc[sell_condition] = -1.0

        # 清除因滚动窗口无法计算初期的脏数据
        signal.loc[vix_z.isna() | jpy_ma60.isna()] = 0.0

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"