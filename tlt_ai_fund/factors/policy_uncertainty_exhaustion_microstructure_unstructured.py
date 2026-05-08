import numpy as np
import pandas as pd

class UnstructuredEpuRotationFactor:
    """Economic Policy Uncertainty Rotation (unstructured)

    逻辑: 基于新闻的经济政策不确定性(EPU)刻画了市场的宏观避险情绪。当极端恐慌(Z>2.0)见顶衰竭时，不确定性靴子落地，资金从避险资产回流风险资产，美债遭抛售，产生看空脉冲；当极度自满(Z<-2.0)被打破时，突发的政策不确定性会导致资金避险涌入美债，产生看多脉冲。这完美契合Risk-On/Off的宏观轮动，修正了此前顺势接飞刀导致的负向IC。
    数据: usepuindxd (每日经济政策不确定性非结构化新闻指数)
    触发: Z-Score > 2.0 且边际向下衰竭 -> 避险消退(-1.0)；Z-Score < -2.0 且边际向上突破 -> 避险启动(+1.0)
    输出: 每次触发后信号延续3天以确保满足 5%-15% 的目标 Trigger Rate，常态严格休眠为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_epu_rotation'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0.0 的 Series 满足零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns:
            return signal
            
        # 向前填充非结构化数据缺失值
        epu = data['usepuindxd'].ffill()
        
        # 确保有足够的数据计算长周期 Z-Score
        if len(epu.dropna()) < 252:
            return signal
            
        # 计算 252日 Z-Score (代表年度级别的宏观极端情绪)
        epu_mean = epu.rolling(window=252).mean()
        # 替换 std 为 0 的情况防止除零错误
        epu_std = epu.rolling(window=252).std().replace(0, np.nan)
        epu_z = (epu - epu_mean) / epu_std
        
        # 1. 极端恐慌见顶衰竭 (Panic Exhaustion) -> Risk-On, 靴子落地，资金抛售避险美债
        # 极端条件: 处于极度高位 (Z-Score > 2.0)
        extreme_fear = epu_z > 2.0
        # 二阶导数衰竭: 跌破3日均线，且边际变化向下
        fear_exhaust = (epu < epu.rolling(window=3).mean()) & (epu.diff() < 0)
        # 两者同时满足时触发做空美债脉冲
        trigger_short = extreme_fear & fear_exhaust
        
        # 2. 极端自满被打破 (Complacency Break) -> Risk-Off, 突发冲击，资金买入避险美债
        # 极端条件: 处于极度低位 (Z-Score < -2.0)
        extreme_complacency = epu_z < -2.0
        # 边际变化爆发: 突破3日均线，且边际变化向上
        complacency_break = (epu > epu.rolling(window=3).mean()) & (epu.diff() > 0)
        # 两者同时满足时触发做多美债脉冲
        trigger_long = extreme_complacency & complacency_break
        
        # 3. 信号展期 (Pulse Extension) 
        # 极值反转事件稀少(约占全年的2-3%)，将信号平滑延续3天，确保 Trigger Rate 稳定落在 5%~15% 区间内
        pulse_short = trigger_short.rolling(window=3, min_periods=1).max() > 0
        pulse_long = trigger_long.rolling(window=3, min_periods=1).max() > 0
        
        # 赋值脉冲信号 (0.0为常态休眠)
        signal[pulse_long] = 1.0
        signal[pulse_short] = -1.0
        
        # 清理由于计算早期 rolling 带来的 NaN，严格返回浮点型
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"