import numpy as np
import pandas as pd

class DualPanicExhaustionFactor:
    """Dual Panic Exhaustion Factor (panic_mean_reversion/nonlinear)

    逻辑: 交叉衡量股市情绪恐慌(VIX)和实体信用恐慌(高收益债利差)。当两者综合Z-Score极高且同时回落时(极度恐慌衰竭)，触发强烈看多；当处于轻度恐慌且VIX短期均线向上发散时(钝刀割肉期)，触发看空。
    数据: [vixcls, bamlh0a0hym2]
    输出: +1.0(恐慌见顶衰竭，抄底美股) / -1.0(轻度恐慌恶化，趋势规避)
    触发条件: 综合恐慌得分 > 1.5 且双指标差分<=0时输出+1.0；0.5 < 得分 <= 1.5 且 VIX > 5日均线且当日上升时输出-1.0，预期 Trigger Rate 约 8%-15%
    """

    def __init__(self):
        self.name = 'dual_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖数据列是否存在
        required_cols = ['vixcls', 'bamlh0a0hym2']
        if not all(col in data.columns for col in required_cols):
            return signal

        # 提取数据并前向填充处理缺失值
        vix = data['vixcls'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()

        # 计算252个交易日(约一年)的滚动Z-Score
        vix_mean = vix.rolling(window=252, min_periods=126).mean()
        vix_std = vix.rolling(window=252, min_periods=126).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std

        hy_mean = hy_spread.rolling(window=252, min_periods=126).mean()
        hy_std = hy_spread.rolling(window=252, min_periods=126).std().replace(0, np.nan)
        hy_z = (hy_spread - hy_mean) / hy_std

        # 构建高维交叉的综合恐慌得分 (等权组合)
        panic_score = (vix_z + hy_z) / 2.0

        # 计算指标的边际变化(一阶导数)
        vix_diff = vix.diff()
        hy_diff = hy_spread.diff()
        
        # 计算VIX的5日(一周)短期趋势
        vix_ma5 = vix.rolling(window=5).mean()

        # 核心逻辑1：二阶导数防飞刀，极值+衰竭 = 强看多 (+1.0)
        # 综合Z-Score > 1.5 (极度恐慌区间)，且VIX和信用利差当天同步停止恶化/开始回落
        buy_cond = (
            (panic_score > 1.5) & 
            (vix_diff < 0) & 
            (hy_diff <= 0)
        ).fillna(False)

        # 核心逻辑2：长牛物理属性，轻微恐慌且正在恶化 = 趋势看空 (-1.0)
        # 综合Z-Score在 0.5 到 1.5 之间(恐慌初期/中期)，且VIX当天继续上升并突破5日均线
        sell_cond = (
            (panic_score > 0.5) & 
            (panic_score <= 1.5) & 
            (vix_diff > 0) & 
            (vix > vix_ma5)
        ).fillna(False)

        # 赋值脉冲信号 (默认0.0休眠)
        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"