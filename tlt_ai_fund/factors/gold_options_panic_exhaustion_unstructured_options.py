import numpy as np
import pandas as pd

class GoldOptionsPanicExhaustionFactor:
    """黄金期权波动率恐慌衰竭 (unstructured/options)

    逻辑: 捕捉黄金期权隐含波动率(GVZCLS)的极端边际飙升与衰竭。黄金期权波动率代表了对无结构通胀冲击或地缘黑天鹅的避险情绪。当此类恐慌边际变化达到极端脉冲(Z-Score>2.5)且开始回落衰竭时，表明宏观恐慌情绪触顶，流动性重置，长端美债(TLT)迎来确定性较高的超跌反弹或均值回归。反之亦然。
    数据: gvzcls (CBOE黄金ETF隐含波动率指数)
    触发: 
      多头(+1.0): GVZCLS 5日变动量的126日 Z-Score > 2.5 (极度恐慌)，且当日回落并跌破3日均线。
      空头(-1.0): GVZCLS 5日变动量的126日 Z-Score < -2.5 (极度自满)，且当日反弹并突破3日均线。
    输出: [-1.0, 1.0] 的狙击手级脉冲信号。
    """

    def __init__(self):
        self.name = 'gold_options_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (常态信号必须为 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        if 'gvzcls' not in data.columns:
            return signal
            
        # 基础数据预处理
        gvz = data['gvzcls'].ffill()
        
        # 铁律3: 边际变化 (绝对禁止直接使用绝对水位)
        # 计算 5 日动量变化，捕捉预期突变瞬间
        gvz_diff5 = gvz.diff(5)
        
        # 计算变动量的 126 日(约半年)滚动 Z-Score
        roll_mean = gvz_diff5.rolling(window=126, min_periods=63).mean()
        roll_std = gvz_diff5.rolling(window=126, min_periods=63).std()
        
        z_score = (gvz_diff5 - roll_mean) / (roll_std + 1e-8)
        
        # 铁律2: 二阶导数 (必须等待极值衰竭，严禁接飞刀)
        # --- 多头信号: 极度恐慌脉冲 + 开始衰竭回落 ---
        extreme_panic = z_score > 2.5
        panic_exhaustion = (gvz.diff(1) < 0) & (gvz < gvz.rolling(window=3).mean())
        
        # --- 空头信号: 极度自满脉冲 + 开始恐慌反弹 ---
        extreme_complacency = z_score < -2.5
        complacency_rebound = (gvz.diff(1) > 0) & (gvz > gvz.rolling(window=3).mean())
        
        # 赋值脉冲信号
        signal[extreme_panic & panic_exhaustion] = 1.0
        signal[extreme_complacency & complacency_rebound] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"