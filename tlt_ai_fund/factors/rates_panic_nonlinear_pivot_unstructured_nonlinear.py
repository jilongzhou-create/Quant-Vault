import numpy as np
import pandas as pd

class RatesPanicNonlinearPivotFactor:
    """Rates Panic Nonlinear Pivot Factor (unstructured/nonlinear)

    逻辑: 捕捉"恐慌极值衰竭"与"降息预期突变"的非线性共振。当市场恐慌(VIX极高)引发流动性危机结束(VIX开始回落)，且前端利率(dgs2)急跌带动收益率曲线急剧牛陡(Bull Steepening)时，确认美联储已被迫转向，此时做多美债。脉冲信号规避了单边恐慌期的现金为王(接飞刀)效应。
    数据: dgs2, t10y2y, vixcls
    触发: VIX Z-Score > 2.0且低于3日均值(衰竭) + dgs2 5日跌幅 Z-Score < -2.0(边际突变) + t10y2y 5日涨幅 Z-Score > 1.5(形态确认)
    输出: +1.0 (强烈看多美债脉冲)，非常态时为 0.0
    """

    def __init__(self):
        self.name = 'rates_panic_nonlinear_pivot'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值信号 (严格遵守铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)

        # 检查所需数据列是否存在
        required_cols = ['dgs2', 't10y2y', 'vixcls']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 向前填充缺失值，保证宏观数据对齐
        df = data[required_cols].ffill()

        # --- 边际变化计算 (严格遵守铁律3: 边际变化) ---
        # 计算 5 个交易日（约一周）的动量变化，捕捉预期突变的瞬间
        dgs2_diff5 = df['dgs2'].diff(5)
        t10y2y_diff5 = df['t10y2y'].diff(5)

        # --- Z-Score 计算 (252个交易日滚动窗口，约一年基准) ---
        # 1. dgs2 边际突跳 Z-Score
        dgs2_diff_mean = dgs2_diff5.rolling(window=252).mean()
        dgs2_diff_std = dgs2_diff5.rolling(window=252).std().replace(0, np.nan)
        dgs2_z = (dgs2_diff5 - dgs2_diff_mean) / dgs2_diff_std

        # 2. t10y2y 边际变陡 Z-Score
        t10y2y_diff_mean = t10y2y_diff5.rolling(window=252).mean()
        t10y2y_diff_std = t10y2y_diff5.rolling(window=252).std().replace(0, np.nan)
        t10y2y_z = (t10y2y_diff5 - t10y2y_diff_mean) / t10y2y_diff_std

        # 3. VIX 绝对水位 Z-Score
        vix_mean = df['vixcls'].rolling(window=252).mean()
        vix_std = df['vixcls'].rolling(window=252).std().replace(0, np.nan)
        vix_z = (df['vixcls'] - vix_mean) / vix_std

        # --- 核心非线性交叉触发条件 ---
        
        # 条件A: 降息预期突跳 (短端利率 dgs2 出现极端两倍标准差下行)
        cond_rate_cut = dgs2_z < -2.0

        # 条件B: 曲线形态确认 (短端下行主导的急剧 Bull Steepening)
        cond_bull_steep = t10y2y_z > 1.5

        # 条件C: 二阶导数衰竭铁律 (严格遵守铁律2: 拒绝接飞刀)
        # 必须同时满足: VIX处于极端高位(>2.0标准差) AND VIX动量已经开始衰竭(跌破3日均线)
        cond_vix_extreme = vix_z > 2.0
        cond_vix_exhaustion = df['vixcls'] < df['vixcls'].rolling(window=3).mean()

        # --- 信号生成 ---
        # 只有在高维非线性条件同时满足的瞬间，才输出狙击手级别的多头脉冲
        trigger = cond_rate_cut & cond_bull_steep & cond_vix_extreme & cond_vix_exhaustion
        
        signal[trigger] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"