import numpy as np
import pandas as pd

class UnstructuredEpuPivotCrossFactor:
    """Unstructured EPU Pivot Cross (unstructured/nonlinear)

    逻辑: 交叉结合文本分析类的经济政策不确定性(EPU)极值衰竭，以及债券短端定价预期的突变。当政策恐慌见顶回落且短端利率显著下行驱动曲线牛陡时，说明避险资金正与降息预期共振，生成看多脉冲。反之生成看空脉冲。
    数据: usepuindxd, dgs2, t10y2y
    触发: usepuindxd Z-Score>1.5且开始回落 + dgs2显著下行(<-0.5*STD) + t10y2y显著变陡 -> +1.0
    输出: 狙击手级别的脉冲信号
    """

    def __init__(self, window: int = 63, epu_z_thresh: float = 1.5):
        self.name = 'unstructured_epu_pivot_cross'
        self.window = window
        self.epu_z_thresh = epu_z_thresh

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['usepuindxd', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        df = data[required_cols].ffill()
        
        # 1. 文本分析类宏观情绪: US Economic Policy Uncertainty Index
        epu = df['usepuindxd']
        epu_mean = epu.rolling(window=self.window, min_periods=10).mean()
        epu_std = epu.rolling(window=self.window, min_periods=10).std()
        
        # 计算 EPU 的 Z-Score，反映预期的异常偏离
        epu_z = (epu - epu_mean) / epu_std.replace(0, 1e-5)
        
        # 遵循二阶导数铁律: 必须等情绪指标出现衰竭拐点才操作，严禁接飞刀
        epu_diff_1 = epu.diff(1)
        
        # 2. 利率预期的边际变化 (动量视角)
        dgs2_diff = df['dgs2'].diff(3)
        dgs2_diff_std = dgs2_diff.rolling(window=self.window, min_periods=10).std()
        t10y2y_diff = df['t10y2y'].diff(3)
        
        # 3. 动态波动率标准化过滤
        # 短端利率显著下行 (降息预期快速发酵)
        dgs2_down = dgs2_diff < -0.5 * dgs2_diff_std
        # 短端利率显著上行 (鹰派预期快速发酵)
        dgs2_up = dgs2_diff > 0.5 * dgs2_diff_std
        
        # 4. 非线性特征交叉
        # 多头脉冲: 政策不确定性处于极高位且单日边际衰退 + 市场实质定价降息(短端暴降且曲线牛陡)
        cond_long = (
            (epu_z > self.epu_z_thresh) & 
            (epu_diff_1 < 0.0) & 
            dgs2_down & 
            (t10y2y_diff > 0.0)
        )
        
        # 空头脉冲: 政策不确定性处于极低盲区且开始抬头 + 市场实质定价加息(短端暴涨且曲线熊平)
        cond_short = (
            (epu_z < -self.epu_z_thresh) & 
            (epu_diff_1 > 0.0) & 
            dgs2_up & 
            (t10y2y_diff < 0.0)
        )
        
        signal[cond_long] = 1.0
        signal[cond_short] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, epu_z_thresh={self.epu_z_thresh})"