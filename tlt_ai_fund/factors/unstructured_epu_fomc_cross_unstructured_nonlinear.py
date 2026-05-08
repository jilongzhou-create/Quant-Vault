import numpy as np
import pandas as pd

class UnstructuredEpuMacroRegimeNonlinearFactor:
    """经济政策不确定性与曲线形态非线性交叉因子 (unstructured/nonlinear)

    逻辑: 结合经济政策不确定性(EPU)与收益率曲线前瞻预期的非线性交叉。捕捉EPU较高且开始边际衰竭时，市场定价降息（短端利率下行+曲线牛陡 Bull Steepening），触发看多美债的脉冲；反之捕捉极度自满(EPU水位偏低)转向上升且美联储超预期紧缩（短端利率上行+曲线熊平 Bear Flattening）的看空脉冲。通过宏观情绪拐点与利率预期的共振提供正交Alpha。
    数据: usepuindxd (经济政策不确定性指数), dgs2 (2年期美债), t10y2y (期限利差)
    触发: EPU Z-Score的绝对值处于极值且边际反转 + dgs2边际方向 + t10y2y边际变化方向同步满足
    输出: [-1.0, 1.0] 狙击手级别的脉冲信号
    """

    def __init__(self, z_threshold=0.6, epu_window=5, z_window=252, diff_window=3):
        self.name = 'unstructured_epu_macro_regime_nonlinear'
        self.z_threshold = z_threshold
        self.epu_window = epu_window
        self.z_window = z_window
        self.diff_window = diff_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['usepuindxd', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 提取并前向填充数据，避免因节假日或发布频率差异导致的数据缺失干扰
        epu = data['usepuindxd'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # EPU的平滑与滚动的年度Z-Score (衡量政策不确定性所处的宏观状态水位)
        epu_ma = epu.rolling(window=self.epu_window).mean()
        epu_ma_mean = epu_ma.rolling(window=self.z_window).mean()
        epu_ma_std = epu_ma.rolling(window=self.z_window).std()
        
        # 避免除以0或极小值
        epu_ma_std = epu_ma_std.replace(0, np.nan)
        epu_z = (epu_ma - epu_ma_mean) / epu_ma_std
        
        # 核心铁律3: 边际变化 (捕捉衰竭与市场预期的动态突变瞬间)
        epu_diff = epu_ma.diff(self.diff_window)
        dgs2_diff = dgs2.diff(self.diff_window)
        t10y2y_diff = t10y2y.diff(self.diff_window)
        
        # 核心铁律2: 二阶导数极值衰竭 + 跨域逻辑共振
        
        # 多头触发脉冲 (降息交易/危机缓解):
        # 1. 不确定性处于相对高位 (Z > threshold)
        # 2. 恐慌情绪开始边际衰退 (epu_diff < 0)
        # 3. 2年期短端收益率下行反映放水预期 (dgs2_diff < 0)
        # 4. 短端下行快于长端，收益率曲线牛陡 (t10y2y_diff > 0)
        bull_cond = (
            (epu_z > self.z_threshold) & 
            (epu_diff < 0) & 
            (dgs2_diff < 0) & 
            (t10y2y_diff > 0)
        )
        
        # 空头触发脉冲 (紧缩冲击/自满被打破): 
        # 1. 市场处于极度自满，不确定性偏低 (Z < -threshold)
        # 2. 不确定性突然飙升 (epu_diff > 0)
        # 3. 2年期短端收益率上行反映紧缩 (dgs2_diff > 0)
        # 4. 短端上行快于长端，收益率曲线熊平 (t10y2y_diff < 0)
        bear_cond = (
            (epu_z < -self.z_threshold) & 
            (epu_diff > 0) & 
            (dgs2_diff > 0) & 
            (t10y2y_diff < 0)
        )
        
        # 核心铁律1: 零值休眠，条件不满足则输出0.0
        signal.loc[bull_cond] = 1.0
        signal.loc[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_threshold={self.z_threshold}, diff_window={self.diff_window})"