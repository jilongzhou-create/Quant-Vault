import numpy as np
import pandas as pd

class UnstructuredEpuVolatilityPivotFactor:
    """政策不确定性波动率极值反转因子 (unstructured/unstructured)

    逻辑: 衡量经济政策不确定性(EPU)的"混乱程度"(波动率)。当政策混乱度极速飙升至极端(Z>2.5)且开始回落时，意味着恐慌见顶避险退潮，看空美债；当政策混乱度极速下降至极度自满且开始反弹时，意味着黑天鹅重新酝酿，看多美债。
    数据: usepuindxd (美国经济政策不确定性指数)
    触发: EPU 21日波动的5日边际变化 Z-Score > 2.5 且开始衰竭回落 -> -1.0; Z-Score < -2.5 且打破自满回升 -> +1.0
    输出: [-1.0, 1.0] 狙击手级脉冲信号
    """

    def __init__(self, vol_window=21, diff_window=5, z_window=252, z_threshold=2.5, exhaust_window=3):
        self.name = 'unstructured_epu_volatility_pivot'
        # 21日代表1个月均交易日，衡量中期混乱度
        self.vol_window = vol_window
        # 5日代表1周动量，捕捉边际突变
        self.diff_window = diff_window
        # 252日代表1年交易日，提供宏观基准水位
        self.z_window = z_window
        self.z_threshold = z_threshold
        # 3日代表短期微观结构，用于确认拐点衰竭
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 基础数据校验
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        epu = data['usepuindxd'].ffill()
        
        # 1. 经济学特征：政策不确定性的二阶衍生 (EPU Volatility)
        # 不关注绝对不确定性多高，而关注不确定性本身的"上蹿下跳"程度
        epu_vol = epu.rolling(window=self.vol_window).std()
        
        # 2. 边际变化铁律：捕捉波动率的边际跳跃动量
        epu_vol_diff = epu_vol.diff(self.diff_window)
        
        # 3. 计算 Z-Score 识别极端脉冲
        roll_mean = epu_vol_diff.rolling(window=self.z_window).mean()
        roll_std = epu_vol_diff.rolling(window=self.z_window).std().replace(0, np.nan)
        epu_vol_diff_z = (epu_vol_diff - roll_mean) / roll_std
        
        # 4. 二阶导数(衰竭)铁律：短期均线判断动量是否枯竭反转
        epu_vol_exhaust_mean = epu_vol.rolling(window=self.exhaust_window).mean()
        
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 触发条件 A：极度混乱后退潮 (看空美债)
        # 极值条件：波动率跳升 > 2.5 个标准差
        # 衰竭条件：波动率开始小于3日均值 (恐慌情绪枯竭)
        short_cond = (epu_vol_diff_z > self.z_threshold) & (epu_vol < epu_vol_exhaust_mean)
        
        # 触发条件 B：极度自满后惊醒 (看多美债)
        # 极值条件：波动率跳水 < -2.5 个标准差 (政策面死水一潭，市场丧失警惕)
        # 衰竭条件：波动率开始大于3日均值 (不确定性微观重燃，避险资金入场)
        long_cond = (epu_vol_diff_z < -self.z_threshold) & (epu_vol > epu_vol_exhaust_mean)
        
        # 安全填充 NaN 避免 pandas 报错
        long_cond = long_cond.fillna(False)
        short_cond = short_cond.fillna(False)
        
        # 赋值脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(vol_window={self.vol_window}, diff_window={self.diff_window}, z_window={self.z_window}, z_threshold={self.z_threshold})"