import numpy as np
import pandas as pd

class EpuGoldVolResonanceFactor:
    """经济政策不确定性与避险期权共振因子 (unstructured/options)

    逻辑: 结合非结构化文本衍生的政策不确定性(EPU)与避险资产的期权波动率(GVZ)。当 EPU 或 黄金隐含波动率的边际变化发生极端向上脉冲时，说明市场遭遇严重宏观/货币信用冲击。随后，当黄金波动率开始回落(期权恐慌衰竭)，说明流动性挤兑最高峰已过，宏观避险资金开始系统性流入底层避险资产(美债TLT)，产生看多脉冲。反之则为看空脉冲。
    数据: usepuindxd (经济政策不确定性指数), gvzcls (CBOE黄金ETF期权隐含波动率)
    触发: 动量变化量(diff(5))的 252 日 Z-Score > 2.5，且黄金波动率(gvzcls)跌破3日均线(二阶导数衰竭)。
    输出: +1.0 (避险资金涌入美债), -1.0 (风险偏好极度复苏)，非触发日严格为 0.0。
    """

    def __init__(self, window=252, diff_days=5, exhaust_days=3, z_threshold=2.5):
        self.name = 'epu_gold_vol_resonance'
        self.window = window
        self.diff_days = diff_days
        self.exhaust_days = exhaust_days
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        if 'usepuindxd' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        epu = data['usepuindxd'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change Only)
        # 捕捉政策预期与期权隐含波动率的变化瞬间
        epu_diff = epu.diff(self.diff_days)
        gvz_diff = gvz.diff(self.diff_days)
        
        # 滚动计算 Z-Score (避免魔法数字，使用经济学视角的1个交易年252日)
        epu_z = (epu_diff - epu_diff.rolling(self.window).mean()) / epu_diff.rolling(self.window).std()
        gvz_z = (gvz_diff - gvz_diff.rolling(self.window).mean()) / gvz_diff.rolling(self.window).std()
        
        # 捕捉任何一端触发的宏观极端冲击事件
        bull_shock = (epu_z > self.z_threshold) | (gvz_z > self.z_threshold)
        bear_shock = (epu_z < -self.z_threshold) | (gvz_z < -self.z_threshold)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 买入美债必须等恐慌情绪(GVZ)明确见顶回落，避免接主跌浪的飞刀
        bull_exhaustion = gvz < gvz.rolling(self.exhaust_days).mean()
        
        # 卖出美债必须等风险偏好狂热情绪(GVZ极低)触底并停止下跌
        bear_exhaustion = gvz > gvz.rolling(self.exhaust_days).mean()
        
        # 组合极值条件与衰竭条件，输出脉冲信号
        bull_cond = bull_shock & bull_exhaustion
        bear_cond = bear_shock & bear_exhaustion
        
        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, diff_days={self.diff_days}, exhaust_days={self.exhaust_days}, z_threshold={self.z_threshold})"