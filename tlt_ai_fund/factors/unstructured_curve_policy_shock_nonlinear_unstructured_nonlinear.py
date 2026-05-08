import numpy as np
import pandas as pd

class UnstructuredEpuMacroPivotFactor:
    """政策不确定性与利率预期交叉反转因子 (unstructured/nonlinear)

    逻辑: 结合非结构化的经济政策不确定性指数(EPU)与2年期美债收益率(衡量政策预期)。当EPU处于极端高位引发市场恐慌, 随后EPU开始回落(衰竭), 且2年期美债收益率同步下行(降息预期突变)时, 形成政策底与情绪底的双重共振, 脉冲看多美债。反之亦然。通过适度放宽极端阈值(Z>0.75, 约前25%分位)并配合3日动量衰竭，确保Trigger Rate处于5%-15%区间。
    数据: usepuindxd (每日经济政策不确定性), dgs2 (2年期美债收益率)
    触发: EPU 5日均值的252日 Z-Score > 0.75 且回落, 叠加 dgs2 下行 -> +1.0
    输出: 狙击手脉冲信号, 捕捉政策与情绪边际突变驱动的美债主升/主跌浪
    """

    def __init__(self):
        self.name = 'unstructured_epu_macro_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 检查必要字段是否存在
        if 'usepuindxd' not in data.columns or 'dgs2' not in data.columns:
            return signal

        # 缺失值前向填充
        epu = data['usepuindxd'].ffill()
        dgs2 = data['dgs2'].ffill()

        # 1. 计算 EPU 5日平滑均线及 252日(1年) Z-Score
        epu_ma = epu.rolling(window=5).mean()
        epu_mean_252 = epu_ma.rolling(window=252).mean()
        epu_std_252 = epu_ma.rolling(window=252).std().replace(0, np.nan)
        epu_z = (epu_ma - epu_mean_252) / epu_std_252

        # 2. 衰竭与边际变化条件 (二阶导数与边际变化铁律)
        # 不确定性情绪的边际衰竭 (3日动量反转)
        epu_exhaust_bull = epu_ma.diff(3) < 0
        epu_exhaust_bear = epu_ma.diff(3) > 0
        
        # dgs2 的边际变化反映货币政策预期的突变
        dgs2_diff_bull = dgs2.diff(3) < 0
        dgs2_diff_bear = dgs2.diff(3) > 0

        # 3. 脉冲触发逻辑
        # 多头：不确定性处于高位(前25%)但开始缓解 + 降息预期升温(短端利率下降)
        buy_cond = (epu_z > 0.75) & epu_exhaust_bull & dgs2_diff_bull
        
        # 空头：不确定性处于低位(后25%)但开始抬头 + 加息预期升温(短端利率上升)
        sell_cond = (epu_z < -0.75) & epu_exhaust_bear & dgs2_diff_bear

        # 赋值狙击手脉冲信号
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"