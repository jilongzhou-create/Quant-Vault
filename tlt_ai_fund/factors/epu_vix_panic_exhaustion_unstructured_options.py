import numpy as np
import pandas as pd

class UnstructuredEpuGvzFearExhaustionFactor:
    """Unstructured Options Macro Fear Exhaustion Factor (unstructured/options)

    逻辑: 结合非结构化经济政策不确定性(EPU)与期权隐含黄金波动率(GVZ)。当两者合成的宏观恐慌指数飙升至极端水平并开始衰竭时，代表不管是通胀冲击还是流动性冲击的最坏时刻已过，避险及期限溢价将压缩，利多美债(TLT)；反之，当恐慌极度低迷且开始抬头时，代表市场打破自满，尾部政策风险重燃，期限溢价扩张，利空美债。
    数据: usepuindxd (非结构化数据), gvzcls (期权黄金波动率)
    触发: 合成恐慌指数 126日 Z-Score > 1.35 且开始回落 (<4日均值) -> 脉冲+1.0；Z-Score < -1.35 且开始反弹 (>4日均值) -> 脉冲-1.0
    输出: 脉冲型，[-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'unstructured_epu_gvz_fear_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值信号 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查基础数据字段是否存在
        if 'usepuindxd' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 1. 数据对齐与前向填充处理缺失值
        epu = data['usepuindxd'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 2. 计算边际变化 (边际变化铁律: 使用一个季度/63日滚动Z-Score衡量相对近期基准的边际突变)
        epu_z = (epu - epu.rolling(63).mean()) / epu.rolling(63).std().replace(0, np.nan)
        gvz_z = (gvz - gvz.rolling(63).mean()) / gvz.rolling(63).std().replace(0, np.nan)
        
        # 合成总宏观恐慌指数 (跨资产共振：政策不确定性 + 避险资产期权波动率)
        shock_idx = epu_z + gvz_z
        
        # 3. 恐慌指数的极端水位识别 (使用半年/126日窗口，避免魔法数字)
        shock_idx_z = (shock_idx - shock_idx.rolling(126).mean()) / shock_idx.rolling(126).std().replace(0, np.nan)
        
        # 4. 衰竭/转折条件 (二阶导数铁律: 绝对禁止直接追高，必须等均线拐点，这里使用近1周/4日均线)
        shock_ma = shock_idx.rolling(4).mean()
        
        # 5. 狙击手脉冲信号生成
        # 多头：宏观恐慌极度高涨 (Z > 1.35 对应约8.8%分位数)，且刚刚开始消退 -> 冲击结束，利好美债
        buy_cond = (shock_idx_z > 1.35) & (shock_idx < shock_ma)
        
        # 空头：宏观极度自满 (Z < -1.35)，且恐慌刚刚开始抬头 -> 冲击开启，利空美债
        sell_cond = (shock_idx_z < -1.35) & (shock_idx > shock_ma)
        
        # 只在触发条件瞬间输出信号
        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"