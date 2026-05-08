import numpy as np
import pandas as pd

class MicrostructureCrossVolReversalFactor:
    """微观结构跨资产波动反转因子 (microstructure/nonlinear)

    逻辑: 跨资产隐含波动率(VIX+GVZ)反映了宏观避险情绪的微观交易结构极值。常态下信号为0；当波动率极高且开始回落(衰竭)，同时美债收益率曲线迅速变陡(确认降息预期)，预示流动性危机解除且货币政策转向，发出看多美债脉冲。反之，当低波动平缓期被打破且曲线急剧走平(定价加息/通胀)，预示货币紧缩重启，发出看空脉冲。
    数据: vixcls, gvzcls, t10y2y
    触发: 联合波动率Z-Score > 1.2 且低于3日均值 且 曲线变陡(diff>0) -> +1.0；Z-Score < -1.0 且高于3日均值 且 曲线走平(diff<0) -> -1.0
    输出: 狙击手级别脉冲信号，非线性识别多空拐点。
    """

    def __init__(self):
        self.name = 'microstructure_cross_vol_reversal_volatility_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查是否包含所需字段
        required_cols = ['vixcls', 'gvzcls', 't10y2y']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return signal

        # 前向填充缺失值以处理节假日不对齐
        df = data[required_cols].ffill()

        # 稳健性处理: 黄金波动率 gvzcls 早期数据如果缺失，以 vixcls 水平等价填充，保证因子连续性
        vix = df['vixcls']
        gvz = df['gvzcls'].fillna(vix)
        
        # 联合跨资产微观波动率水平
        cross_vol = vix + gvz
        
        # 126个交易日 (约半年) 滚动 Z-Score，反映局部中周期的非线性极值
        window = 126
        vol_mean = cross_vol.rolling(window).mean()
        vol_std = cross_vol.rolling(window).std().replace(0, np.nan)
        cross_vol_z = (cross_vol - vol_mean) / vol_std

        # 铁律2: 二阶导数判断 (绝不接飞刀，必须出现拐点衰竭)
        vol_ma3 = cross_vol.rolling(3).mean()
        vol_exhaustion_down = cross_vol < vol_ma3  # 恐慌开始衰竭回落
        vol_reversal_up = cross_vol > vol_ma3      # 极度自满期被打破，波动率抬头

        # 铁律3: 边际动量确认 (收益率曲线边际变化)
        # 陡峭化(>0)配合恐慌衰竭形成多头，扁平化(<0)配合波动反弹形成空头 (巧妙规避2022单边主跌浪)
        curve_momentum = df['t10y2y'].diff(3)

        # 核心触发逻辑
        long_cond = (cross_vol_z > 1.2) & vol_exhaustion_down & (curve_momentum > 0.0)
        short_cond = (cross_vol_z < -1.0) & vol_reversal_up & (curve_momentum < 0.0)

        # 仅在触发日赋值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"