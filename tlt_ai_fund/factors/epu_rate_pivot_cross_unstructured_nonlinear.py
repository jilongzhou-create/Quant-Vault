import numpy as np
import pandas as pd

class EpuRatePivotCrossFactor:
    """政策不确定性与降息预期非线性共振脉冲 (unstructured/nonlinear)

    逻辑: 挖掘非结构化新闻指数(经济政策不确定性 EPU)与前瞻利率的交叉共振。
          高不确定性期间直接买入长债可能遭遇流动性危机的"股债双杀"。必须等待恐慌极值见顶衰竭, 
          且伴随短端利率剧烈下行与曲线变陡(确认美联储宽松转向发酵)时, 才触发长债的多头狙击脉冲。
    数据: usepuindxd (经济政策不确定性), dgs2 (2年期国债收益率), t10y2y (10年-2年利差)
    触发: EPU Z-Score > 1.25 且回落(衰竭) + DGS2 变化量 Z-Score < -1.25 (急剧下行) + t10y2y 边际变陡。
    输出: +1.0 看多美债(恐慌顶+降息发酵), -1.0 看空美债(平稳破裂+加息发酵), 严守零值休眠。
    """

    def __init__(self):
        self.name = 'epu_rate_pivot_cross'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠, 初始信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['usepuindxd', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 填充缺失值并保持时序逻辑
        epu = data['usepuindxd'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 计算 EPU 的极值状态 (60日滚动窗口, 反映中期政策情绪水位)
        epu_z = (epu - epu.rolling(60).mean()) / epu.rolling(60).std()
        epu_diff = epu.diff(3)  # 铁律3: 边际变化
        
        # 铁律3: 边际变化 (对利率类预期指标严格使用动量变化而非绝对水位)
        dgs2_diff = dgs2.diff(3)
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(60).mean()) / dgs2_diff.rolling(60).std()
        
        t10y2y_diff = t10y2y.diff(3)
        
        # 铁律2: 二阶导数与衰竭防飞刀
        # 做多脉冲: 不确定性恐慌见顶衰竭 + 短端利率急剧边际下行 + 曲线剧烈变陡(Bull Steepening)
        long_cond = (
            (epu_z > 1.25) & (epu_diff < 0) & 
            (dgs2_z < -1.25) & (dgs2_diff < 0) & 
            (t10y2y_diff > 0)
        )
        
        # 做空脉冲: 政策平稳期被打破恶化 + 短端利率急剧边际飙升 + 曲线平坦或倒挂(Bear Flattening)
        short_cond = (
            (epu_z < -1.25) & (epu_diff > 0) & 
            (dgs2_z > 1.25) & (dgs2_diff > 0) & 
            (t10y2y_diff < 0)
        )
        
        # 只在多空共振节点输出离散脉冲
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"