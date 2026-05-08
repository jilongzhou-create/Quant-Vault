import numpy as np
import pandas as pd

class UnstructuredEpuPanicExhaustionFactor:
    """新闻政策不确定性(EPU)极值衰竭因子 (panic_mean_reversion/unstructured)

    逻辑: 捕捉基于新闻报道的经济政策不确定性指数(usepuindxd)的恐慌极值与动量反转。
          美股市场具有极强的"不确定性落地即利好"属性。当政策不确定性飙升至极高位(Z-score > 1.5)
          且绝对值开始回落下穿短期均线时，意味着宏观恐慌衰竭，构成高胜率的抄底买点(+1.0)。
          相反，当不确定性从平静期(Z < 0)开始连续数日稳步上升，突破均值进入轻度恐慌状态时，
          预示市场预期正在发生隐性恶化，进入"钝刀割肉"的震荡下行期，构成卖点(-1.0)。
    数据: usepuindxd (日常经济政策不确定性指数 - 非结构化新闻文本数据)
    输出: 信号范围 [-1.0, 1.0]。+1.0为看多脉冲(买入恐慌衰竭)，-1.0为看空脉冲(预警钝刀割肉)，平时为0.0。
    触发条件: EPU极值+下穿均线/两日连跌(买入)；EPU从低位转为连续3日上升且突破历史均值(卖出)。预期 Trigger Rate 控制在 8% 左右。
    """

    def __init__(self):
        self.name = 'unstructured_epu_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据存在性检查
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index)

        epu = data['usepuindxd'].ffill()
        
        # 避免全 NaN 导致报错
        if epu.isna().all():
            return pd.Series(0.0, index=data.index)
            
        signal = pd.Series(0.0, index=data.index)
        
        # 1. 计算 5 日均线以平滑单日新闻波动的噪音 (1周自然平滑)
        epu_ma5 = epu.rolling(window=5, min_periods=1).mean()
        
        # 2. 计算 252 日(约1个交易年)的 Z-Score，衡量不确定性的长周期相对极值
        epu_mean252 = epu_ma5.rolling(window=252, min_periods=60).mean()
        epu_std252 = epu_ma5.rolling(window=252, min_periods=60).std()
        epu_z = (epu_ma5 - epu_mean252) / (epu_std252 + 1e-6)
        
        # --- 强烈看多信号 (+1.0)：极度恐慌见顶衰竭 (狙击手脉冲) ---
        
        # 衰竭条件A: 前一日 EPU 均线处于极高位(Z > 1.5)，今日 EPU 绝对值清晰下穿 5 日均线
        buy_cond_A = (epu_z.shift(1) > 1.5) & (epu < epu_ma5) & (epu.shift(1) >= epu_ma5.shift(1))
        
        # 衰竭条件B: EPU 处于偏高位(Z > 1.0)，且出现连续两日的绝对值下跌 (二阶导确立反转)
        epu_diff = epu.diff()
        buy_cond_B = (epu_z > 1.0) & (epu_diff < 0) & (epu_diff.shift(1) < 0) & (epu_diff.shift(2) >= 0)
        
        buy_pulse = buy_cond_A | buy_cond_B
        
        # --- 趋势看空信号 (-1.0)：轻度恐慌的"钝刀割肉" (前瞻性预警) ---
        
        # 恶化条件: EPU 均线从平静期(Z < 0)开始，连续 3 日稳步爬升，刚刚迈入轻微恐慌区间(0 < Z < 1.0)
        epu_ma5_diff = epu_ma5.diff()
        epu_rising_3d = (epu_ma5_diff > 0) & (epu_ma5_diff.shift(1) > 0) & (epu_ma5_diff.shift(2) > 0)
        
        # 只在达成连续3日上升的当天触发一击脉冲，防范连续发出看空信号
        sell_pulse = (epu_rising_3d & 
                      (~epu_rising_3d.shift(1)) & 
                      (epu_z > 0.0) & 
                      (epu_z < 1.0) & 
                      (epu_z.shift(3) < 0.0))
        
        # 写入信号
        signal[buy_pulse] = 1.0
        signal[sell_pulse] = -1.0
        
        # 防护: 避免同一天逻辑冲突触发，直接置0
        conflict = buy_pulse & sell_pulse
        signal[conflict] = 0.0
        
        signal.name = self.name
        return signal.fillna(0.0)