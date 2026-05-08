import numpy as np
import pandas as pd

class UnstructuredCrossAssetVolReversalFactor:
    """跨资产波动率极值与新闻不确定性反转因子 (volatility/unstructured)

    逻辑: 当股市波动率(VIX)、抗通胀资产波动率(GVZ)以及基于新闻文本驱动的经济政策不确定性(EPU)同步达到极端高位时，表明跨资产看空与恐慌情绪极度拥挤。一旦这些波动率指标开始回落(二阶导数为负，即边际恐慌消退)，风险平价等系统性对冲策略的空头平仓与重新加杠杆将引发美债(TLT)的剧烈反弹。反之，当综合波动率处于极端低位(极度自满)且开始反弹时，系统性降杠杆将导致美债遭遇无差别抛售。此因子严守狙击手脉冲要求，常态下处于休眠状态。
    数据: usepuindxd (经济政策不确定性新闻指数), vixcls (标普VIX), gvzcls (黄金波动率指数)
    触发: 
      做多脉冲(+1.0): 综合 Z-Score > 1.75 (极端恐慌) 且 综合恐慌指数 < 3日均值 (恐慌衰竭/反转确认)
      做空脉冲(-1.0): 综合 Z-Score < -1.25 (极度自满) 且 综合恐慌指数 > 3日均值 (边际恐慌滋生)
    输出: +1.0 (看多美债脉冲), -1.0 (看空美债脉冲), 0.0 (常态休眠)
    """

    def __init__(self):
        self.name = 'unstructured_cross_asset_vol_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化严格为0.0的脉冲信号 (铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据列是否存在，缺失则直接返回全0序列
        required_cols = ['usepuindxd', 'vixcls', 'gvzcls']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return signal
            
        # 前向填充缺失值以处理可能的节假日不对齐
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算 252 日(1个交易年)滚动 Z-Score，加入 1e-6 防止除以零
        epu_z = (epu - epu.rolling(252).mean()) / (epu.rolling(252).std() + 1e-6)
        vix_z = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-6)
        gvz_z = (gvz - gvz.rolling(252).mean()) / (gvz.rolling(252).std() + 1e-6)
        
        # 构建跨资产与非结构化文本(EPU)的综合恐慌指数 
        # 使用 mean 忽略早年间 GVZ 数据缺失的情况，确保历史完整性
        z_df = pd.concat([epu_z, vix_z, gvz_z], axis=1)
        panic_idx = z_df.mean(axis=1)
        
        # 计算综合恐慌指数的动量均值 (铁律3: 边际变化)
        panic_idx_ma = panic_idx.rolling(3).mean()
        
        # 触发条件 1: 极值拥挤且开始瓦解 (铁律2: Anti-Catch-Falling-Knife)
        # 综合恐慌度极高 (Z > 1.75，约前4%分位) 且 边际恐慌正在消退 (指数回落跌穿3日均线)
        bull_pulse = (panic_idx > 1.75) & (panic_idx < panic_idx_ma)
        
        # 触发条件 2: 极度自满且开始惊升 
        # 综合恐慌度极低 (Z < -1.25，约后10%分位) 且 边际恐慌正在滋生 (指数抬头升穿3日均线)
        bear_pulse = (panic_idx < -1.25) & (panic_idx > panic_idx_ma)
        
        # 赋值狙击手级脉冲信号 (目标 Trigger Rate 锁定在 5% - 15% 区间)
        signal[bull_pulse] = 1.0
        signal[bear_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"