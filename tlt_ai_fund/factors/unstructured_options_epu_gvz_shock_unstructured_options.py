import numpy as np
import pandas as pd

class UnstructuredOptionsEpuGvzShockFactor:
    """经济政策不确定性与黄金波动率共振衰竭因子 (unstructured/options)

    逻辑: 结合非结构化新闻情绪(EPU指数)与避险资产期权波动率(GVZ)。当新闻不确定性或黄金波动率的边际变化出现极端跳跃(Z>2.5)，表明宏观恐慌达到极值。若随后这两者同时开始回落(二阶导衰竭)，说明跨资产恐慌消退，市场开始Price-in宽松预期与流动性修复，此时做多美债(TLT)。
    数据: usepuindxd (经济政策不确定性), gvzcls (黄金ETF隐含波动率)
    触发: (EPU 5日变化 Z-Score > 2.5 或 GVZ 5日变化 Z-Score > 2.5) + 两者当日均跌破3日均线(衰竭确认)
    输出: +1.0 (恐慌衰竭看多)，-1.0 (自满衰竭看空)，常态=0.0，脉冲最长持续3天以满足Trigger Rate
    """

    def __init__(self, z_window=63, change_window=5, smooth_window=3):
        self.name = 'unstructured_options_epu_gvz_shock'
        self.z_window = z_window          # 63天约一个季度，用于计算具有自适应性的滚动 Z-Score
        self.change_window = change_window # 5天刻画边际变化(Marginal Change)
        self.smooth_window = smooth_window # 3天均线用于判定拐点(Anti-Catch-Falling-Knife)

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态下信号必须休眠为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        # 铁律3: 边际变化 (Marginal Change Only) - 严禁直接使用绝对值
        # EPU 为高频新闻指数，先进行轻微平滑过滤日内噪音，再取 5 日差分
        epu_smoothed = data['usepuindxd'].rolling(self.smooth_window).mean()
        epu_change = epu_smoothed.diff(self.change_window)
        gvz_change = data['gvzcls'].diff(self.change_window)
        
        # 计算滚动的 Z-Score 捕捉局部突变瞬间
        epu_change_z = (epu_change - epu_change.rolling(self.z_window).mean()) / epu_change.rolling(self.z_window).std()
        gvz_change_z = (gvz_change - gvz_change.rolling(self.z_window).mean()) / gvz_change.rolling(self.z_window).std()
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife) - 绝对禁止单一边边际极值直接入场
        # 必须等待当日数值跌破短期均线，表明脉冲已经完成寻顶并拐头向下
        epu_exhaustion_bull = data['usepuindxd'] < data['usepuindxd'].rolling(self.smooth_window).mean()
        gvz_exhaustion_bull = data['gvzcls'] < data['gvzcls'].rolling(self.smooth_window).mean()
        
        # 空头信号的衰竭条件：从极度自满(低波动)中拐头向上
        epu_exhaustion_bear = data['usepuindxd'] > data['usepuindxd'].rolling(self.smooth_window).mean()
        gvz_exhaustion_bear = data['gvzcls'] > data['gvzcls'].rolling(self.smooth_window).mean()
        
        # 触发条件组合
        # 只要有一方出现极端恐慌脉冲，且双方都确认衰竭，即触发看多
        extreme_panic = (epu_change_z > 2.5) | (gvz_change_z > 2.5)
        panic_exhausted = epu_exhaustion_bull & gvz_exhaustion_bull
        
        extreme_complacency = (epu_change_z < -2.5) | (gvz_change_z < -2.5)
        complacency_exhausted = epu_exhaustion_bear & gvz_exhaustion_bear
        
        # 生成狙击手级脉冲点
        base_pulse = pd.Series(0.0, index=data.index)
        base_pulse[extreme_panic & panic_exhausted] = 1.0
        base_pulse[extreme_complacency & complacency_exhausted] = -1.0
        
        # 将脉冲点向前适度保持 2 天（总生存期 3 天），以确保目标 Trigger Rate 落入 5%~15% 窗口
        # 使用 replace 保留 +1/-1 后执行 ffill，完美屏蔽所有连续态噪音
        signal = base_pulse.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_window={self.z_window}, change_window={self.change_window})"