import numpy as np
import pandas as pd

class UnstructuredEpuYieldSqueezeFactor:
    """政策不确定性与收益率极值挤压因子 (unstructured/nonlinear)

    逻辑: 结合基于新闻文本构建的经济政策不确定性指数(usepuindxd)与前瞻性政策利率指标(dgs2)，构建非线性的“政策高压共振极值”。当不确定性与短端利率同时飙升至极端高位，随后在边际上开始双双回落时，标志着市场对政策紧缩的恐慌情绪耗竭，触发抄底美债的做多脉冲；反之做空。
    数据: dgs2, usepuindxd
    触发: 联合 Z-Score > 1.5 (极值条件) 且 合并得分3日动量 < 0 且 dgs2 动量 < 0 (衰竭反转条件)。
    输出: 狙击手级别的脉冲信号，非触发日严格为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_epu_yield_squeeze'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号为全 0.0，严格遵守零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        if 'dgs2' not in data.columns or 'usepuindxd' not in data.columns:
            return signal
            
        # 前向填充确保缺失日逻辑连续性
        dgs2 = data['dgs2'].ffill()
        epu = data['usepuindxd'].ffill()
        
        # 计算 60 日滚动 Z-Score (反映近一个季度的相对极值状态)
        dgs2_z = (dgs2 - dgs2.rolling(60).mean()) / dgs2.rolling(60).std()
        epu_z = (epu - epu.rolling(60).mean()) / epu.rolling(60).std()
        
        # 特征交叉：非线性叠加构建联合压力指数 (两个Z-Score的加和，方差被放大)
        combined_score = dgs2_z + epu_z
        
        # 边际变化：计算 3 日动量，捕捉情绪预期的边际衰竭瞬间
        score_mom = combined_score.diff(3)
        dgs2_mom = dgs2.diff(3)
        
        # 核心铁律执行:
        # 1. 极值过滤: > 1.5 确保只抓尾部 ~15% 的极端时刻
        # 2. 二阶导数反飞刀: 绝对禁止直接追高，必须等待动量指标 < 0 (确立趋势衰竭转折)
        
        # 做多脉冲: 恐慌抛售美债的高峰已过，买入
        long_cond = (
            (combined_score > 1.5) & 
            (score_mom < 0) & 
            (dgs2_mom < 0)
        )
        
        # 做空脉冲: 极度乐观与低利率共振被打破，利率开始反弹，做空美债
        short_cond = (
            (combined_score < -1.5) & 
            (score_mom > 0) & 
            (dgs2_mom > 0)
        )
        
        # 赋值触发脉冲
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"