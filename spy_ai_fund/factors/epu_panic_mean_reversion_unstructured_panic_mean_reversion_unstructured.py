import numpy as np
import pandas as pd

class EpuPanicMeanReversionUnstructuredFactor:
    """经济政策不确定性(EPU)极值衰竭脉冲因子 (panic_mean_reversion/unstructured)

    逻辑: 每日新闻经济政策不确定性指数(usepuindxd)是非结构化文本情绪的量化指标。
          当不确定性飙升至极值(Z-Score > 1.8)，极度恐慌发生，必须等待其开始回落(今日diff<0且低于3日均值)时确认恐慌衰竭，触发强看多买入(抄底极点)；
          当不确定性处于中高水位(0.5 < Z <= 1.8)且正在加速恶化(连续上升且3日激增超过0.5倍标准差)时，说明钝刀割肉或风险正在发酵，触发看空脉冲避免接飞刀。
    数据: usepuindxd (Daily US Economic Policy Uncertainty Index)
    输出: [-1.0, 1.0] 脉冲信号。常态为 0.0。
    触发条件: 极值+衰竭时输出 +1.0，中高位加速恶化时输出 -1.0。预期 Trigger Rate 约 5% ~ 10%。
    """

    def __init__(self):
        self.name = 'epu_panic_mean_reversion_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 缺少依赖数据则休眠
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)
            
        epu = data['usepuindxd'].ffill()
        
        # 计算基于一年的长期状态位 (252个交易日)
        epu_mean = epu.rolling(window=252, min_periods=126).mean()
        epu_std = epu.rolling(window=252, min_periods=126).std()
        
        # 防止除零
        epu_std = epu_std.replace(0, np.nan)
        epu_z = (epu - epu_mean) / epu_std
        
        # 计算近期边际变化和趋势
        diff1 = epu.diff(1)
        diff3 = epu.diff(3)
        ma3 = epu.rolling(window=3).mean()
        ma5 = epu.rolling(window=5).mean()
        
        signal = pd.Series(0.0, index=data.index)
        
        # 1. 【恐慌抄底】：二阶导数铁律，极高水位(Z > 1.8)必须等衰竭(当日回落且跌破3日均线)才出手
        buy_cond = (epu_z > 1.8) & (diff1 < 0) & (epu < ma3)
        
        # 2. 【避险看空】：中等恐慌水位(0.5 < Z <= 1.8)且边际加速恶化(近日持续上升，3日增幅巨大)，主跌浪进行中
        sell_cond = (epu_z > 0.5) & (epu_z <= 1.8) & (diff1 > 0) & (epu > ma5) & (diff3 > epu_std * 0.5)
        
        # 灌入脉冲信号
        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"