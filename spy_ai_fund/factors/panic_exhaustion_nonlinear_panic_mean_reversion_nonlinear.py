import numpy as np
import pandas as pd

class PanicExhaustionNonlinearFactor:
    """恐慌极值与均值回归 (panic_mean_reversion/nonlinear)

    逻辑: 结合美股期权隐含波动率(VIX)与高收益债信用利差(HY OAS)进行非线性特征交叉。当任一维度的恐慌指标达到历史极端高位(Z-Score > 1.5), 且两者的短期动量均由正转负(恐慌见顶衰竭), 捕捉"极值+转折"瞬间, 触发抄底买入信号(+1.0)。若风险指标处于温和区间(Z-Score在0.5至1.5)且两者呈共振上升态势, 属于钝刀割肉的趋势恶化, 触发看空信号(-1.0)。
    数据: vixcls, bamlh0a0hym2
    输出: +1.0 (极端恐慌见顶回落, 看多), -1.0 (风险温和发酵, 看空), 0.0 (常态休眠)
    触发条件: 极值+衰竭交叉触发多头脉冲, 温和共振上升触发空头脉冲。预期 Trigger Rate 约 6%-10%。
    """

    def __init__(self, zscore_window: int = 252, extreme_z: float = 1.5, mild_z: float = 0.5, diff_window: int = 3):
        self.name = 'panic_exhaustion_nonlinear'
        self.zscore_window = zscore_window
        self.extreme_z = extreme_z
        self.mild_z = mild_z
        self.diff_window = diff_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查依赖数据是否存在
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        vix = data['vixcls'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()

        # 计算滚动 Z-Score 识别极值状态
        min_p = self.zscore_window // 2
        
        vix_mean = vix.rolling(window=self.zscore_window, min_periods=min_p).mean()
        vix_std = vix.rolling(window=self.zscore_window, min_periods=min_p).std().replace(0, 1e-6)
        vix_z = (vix - vix_mean) / vix_std

        hy_mean = hy_spread.rolling(window=self.zscore_window, min_periods=min_p).mean()
        hy_std = hy_spread.rolling(window=self.zscore_window, min_periods=min_p).std().replace(0, 1e-6)
        hy_z = (hy_spread - hy_mean) / hy_std

        # 计算短期动量识别衰竭或恶化转折点 (二阶导数铁律)
        vix_diff = vix.diff(self.diff_window)
        hy_diff = hy_spread.diff(self.diff_window)

        # 初始化脉冲信号 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)

        # 核心多头逻辑: 至少有一个处于极端恐慌，且两者都在今天出现回落衰竭 (极值 + 衰竭)
        buy_cond = ((vix_z > self.extreme_z) | (hy_z > self.extreme_z)) & (vix_diff < 0) & (hy_diff < 0)

        # 核心空头逻辑: 两者都处于温和恐慌区间，且都在共振上升 (钝刀割肉，非极值但恶化)
        short_cond = (vix_z > self.mild_z) & (vix_z <= self.extreme_z) & \
                     (hy_z > self.mild_z) & (hy_z <= self.extreme_z) & \
                     (vix_diff > 0) & (hy_diff > 0)

        # 生成脉冲信号
        signal[buy_cond] = 1.0
        signal[short_cond] = -1.0

        # 处理缺失值
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, extreme_z={self.extreme_z}, diff_window={self.diff_window})"