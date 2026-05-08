import numpy as np
import pandas as pd

class PanicMeanReversionVixHyCrossFactor:
    """恐慌极值与信用利差非线性交叉因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合VIX与高收益债信用利差(HY Spread)交叉验证恐慌衰竭。极度恐慌产生的是买点，但绝对禁止直接接飞刀；必须等VIX与利差均处于极值(Z-Score极端)，且当日动量转负(恐慌开始回落)时，才输出看多脉冲(+1.0)。反之，当两者处于轻度恐慌状态(0.5~1.5)且缓慢上升时，属于美股的"钝刀割肉"主跌浪，输出看空脉冲(-1.0)。
    数据: vixcls, bamlh0a0hym2
    输出: +1.0 (极度恐慌且衰竭，抄底), -1.0 (轻度恐慌且恶化，趋势看空), 0.0 (常态休眠)
    触发条件: VIX Z-Score > 1.5且单日回落，并伴随利差 Z-Score > 1.0且回落时输出+1.0；VIX与利差在轻度区间且连续2日上升时输出-1.0。预期Trigger Rate 8%-12%。
    """

    def __init__(self, window: int = 63, vix_extreme_z: float = 1.5, hy_extreme_z: float = 1.0):
        self.name = 'panic_mean_reversion_vix_hy_cross'
        self.window = window
        self.vix_extreme_z = vix_extreme_z
        self.hy_extreme_z = hy_extreme_z

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 信号
        signal = pd.Series(0.0, index=data.index)
        
        # 检查是否包含所需列
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return signal

        # 填补周末及节假日造成的短期缺失 (最大3天)
        vix = data['vixcls'].ffill(limit=3)
        hy = data['bamlh0a0hym2'].ffill(limit=3)

        # 计算经济学周期的Z-Score (63天约为一个季度)
        vix_mean = vix.rolling(window=self.window, min_periods=self.window // 2).mean()
        vix_std = vix.rolling(window=self.window, min_periods=self.window // 2).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)

        hy_mean = hy.rolling(window=self.window, min_periods=self.window // 2).mean()
        hy_std = hy.rolling(window=self.window, min_periods=self.window // 2).std()
        hy_z = (hy - hy_mean) / hy_std.replace(0, np.nan)

        # 计算边际动量变化 (二阶导数铁律: 用于验证恐慌是加剧还是衰竭)
        vix_diff1 = vix.diff(1)
        hy_diff1 = hy.diff(1)
        
        vix_diff2 = vix.diff(2)
        hy_diff2 = hy.diff(2)

        # 触发条件 1: 极度恐慌 + 衰竭时刻 = 强烈看多 (+1.0)
        # 逻辑: VIX 和 HY 利差均处于极值(证明市场极度悲观)，但今日开始回落(飞刀落地，恐慌衰竭)
        buy_cond = (
            (vix_z > self.vix_extreme_z) & 
            (vix_diff1 < 0) & 
            (hy_z > self.hy_extreme_z) & 
            (hy_diff1 <= 0)
        )

        # 触发条件 2: 轻度恐慌 + 趋势恶化 = 看空 (-1.0)
        # 逻辑: VIX 和 HY 利差高于均值但尚未引发极度恐慌(钝刀割肉期)，且连续2日处于上升趋势
        sell_cond = (
            (vix_z > 0.5) & (vix_z < self.vix_extreme_z) & 
            (vix_diff2 > 0) & (vix_diff1 > 0) &
            (hy_z > 0.0) & (hy_z < self.hy_extreme_z) & 
            (hy_diff2 > 0) & (hy_diff1 > 0)
        )

        # 赋值脉冲信号
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, vix_z={self.vix_extreme_z}, hy_z={self.hy_extreme_z})"