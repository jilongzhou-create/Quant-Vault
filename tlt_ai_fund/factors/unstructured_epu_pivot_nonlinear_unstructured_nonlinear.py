import numpy as np
import pandas as pd

class UnstructuredEpuPivotNonlinearFactor:
    """政策不确定性反转交叉因子 (unstructured/nonlinear)

    逻辑: 捕捉每日经济政策不确定性(EPU)的极端脉冲与见顶回落，同时通过前瞻政策利率(dgs2)的大幅下行和收益率曲线(t10y2y)的边际变陡来交叉确认降息及避险预期。由于是左侧捕捉恐慌反转点，属于极佳的脉冲狙击信号。
    数据: usepuindxd (经济政策不确定性指数), dgs2 (2年期国债收益率), t10y2y (期限利差)
    触发: EPU 5日变化量 Z-Score > 1.5 且开始回落 (衰竭) + dgs2 5日变化量 Z-Score < -1.5 + 曲线边际变陡 (t10y2y_diff > 0)
    输出: 狙击手级脉冲信号。极端避险且降息共振时输出 +1.0，极端紧缩预期且情绪消散时输出 -1.0，其余时间 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_epu_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须严格为 0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据字段是否存在
        req_cols = ['usepuindxd', 'dgs2', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 填充缺失值，避免前向偏误
        epu = data['usepuindxd'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # --- 铁律3: 边际变化 (Marginal Change Only) ---
        # 严禁使用绝对值，只计算动量变化
        epu_diff5 = epu.diff(5)
        dgs2_diff5 = dgs2.diff(5)
        t10y2y_diff5 = t10y2y.diff(5)
        
        # 计算 Z-Score，使用过去252个交易日滚动窗口，防止 look-ahead bias
        epu_mean = epu_diff5.rolling(window=252, min_periods=60).mean()
        epu_std = epu_diff5.rolling(window=252, min_periods=60).std()
        epu_z = (epu_diff5 - epu_mean) / epu_std
        
        dgs2_mean = dgs2_diff5.rolling(window=252, min_periods=60).mean()
        dgs2_std = dgs2_diff5.rolling(window=252, min_periods=60).std()
        dgs2_z = (dgs2_diff5 - dgs2_mean) / dgs2_std
        
        # --- 铁律2: 二阶导数 (Anti-Catch-Falling-Knife) ---
        # 恐慌见顶衰竭：EPU经历飙升后，当日不再恶化（日级别边际回落）
        epu_exhaustion_long = epu.diff(1) < 0
        # 乐观见底反弹：EPU经历暴跌后，当日不再改善（日级别边际反弹）
        epu_exhaustion_short = epu.diff(1) > 0
        
        # --- 触发条件组装 (非线性特征交叉) ---
        # 多头脉冲：政策恐慌极值 + 开始回落(不接飞刀) + 短端急降预期降息 + 曲线牛陡(Bull Steepening)
        long_cond = (
            (epu_z > 1.5) & 
            epu_exhaustion_long & 
            (dgs2_z < -1.5) & 
            (t10y2y_diff5 > 0)
        )
        
        # 空头脉冲：政策恐慌极度消散 + 开始反弹 + 短端急升预期加息 + 曲线熊平(Bear Flattening)
        short_cond = (
            (epu_z < -1.5) & 
            epu_exhaustion_short & 
            (dgs2_z > 1.5) & 
            (t10y2y_diff5 < 0)
        )
        
        # --- 铁律1: 零值休眠 (Sniper Pulse) ---
        # 只在触发当日释放 +1.0 或 -1.0
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"