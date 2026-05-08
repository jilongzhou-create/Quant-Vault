import numpy as np
import pandas as pd

class UnstructuredPolicyPanicNonlinearFactor:
    """非结构化政策恐慌非线性交叉因子 (unstructured/nonlinear)

    逻辑: 捕捉非结构化数据(经济政策不确定性指数 EPU)的极端脉冲事件。当 EPU 极度飙升并开始衰竭时，标志着宏观恐慌的极端转折。此时绝不能盲目抄底，必须结合前瞻短端利率(dgs2)动量进行非线性交叉：若短端利率开始急跌，确认避险与鸽派降息预期主导，买入美债(脉冲)；若短端利率急升，确认为紧缩与通胀恐慌，做空美债(脉冲)。这是脉冲因子，仅在预期逆转瞬间开火。
    数据: usepuindxd (经济政策不确定性指数), dgs2 (2年期美债收益率)
    触发: usepuindxd 5日变动的 120日 Z-Score > 2.5，且当日回落(衰竭)，交叉 dgs2 的 3日边际变化方向。
    输出: +1.0 (避险/降息确立看多TLT), -1.0 (紧缩/滞胀看空TLT), 0.0 (常态休眠)
    """

    def __init__(self, zscore_window: int = 120, zscore_threshold: float = 2.5):
        self.name = 'unstructured_policy_panic_nonlinear'
        self.zscore_window = zscore_window
        self.zscore_threshold = zscore_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号全为 0.0 (铁律1: 零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['usepuindxd', 'dgs2']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        epu = data['usepuindxd'].ffill()
        dgs2 = data['dgs2'].ffill()
        
        # 1. 核心铁律3：边际变化 (Marginal Change Only)
        # 计算 EPU 的短期飙升动量 (5日变化)，禁止直接输出绝对水位
        epu_momentum = epu.diff(5)
        
        # 计算 Z-Score，衡量边际突变是否达到极端水平
        epu_mom_mean = epu_momentum.rolling(self.zscore_window).mean()
        epu_mom_std = epu_momentum.rolling(self.zscore_window).std()
        epu_zscore = (epu_momentum - epu_mom_mean) / epu_mom_std.replace(0, np.nan)
        
        # 2. 核心铁律2：二阶导数衰竭 (Anti-Catch-Falling-Knife)
        # 条件A: 动量突变极值
        is_epu_extreme = epu_zscore > self.zscore_threshold
        
        # 条件B: 极端情绪必须见顶衰竭，严禁在情绪继续恶化时接飞刀
        # 当日回落 且 跌破最近3日均值
        is_epu_exhausted = (epu.diff(1) < 0) & (epu < epu.rolling(3).mean())
        
        # 3. 方法C：非线性特征交叉 (Cross-Asset Nonlinear Filter)
        # 即使恐慌见顶，也必须有实体经济资产的真实定价来验证，这里使用对政策最敏感的短端美债(dgs2)
        # 使用 dgs2 的 3日边际变化来过滤真实的宏观驱动主线
        dgs2_momentum = dgs2.diff(3)
        
        # 短端利率下跌 -> 避险买盘进入 or 降息预期确认 -> 利好长债 (Bullish)
        is_bullish_confirm = (dgs2_momentum < 0) & (dgs2.diff(1) < 0)
        
        # 短端利率上升 -> 通胀紧缩恐慌 or 加息预期确认 -> 利空长债 (Bearish)
        is_bearish_confirm = (dgs2_momentum > 0) & (dgs2.diff(1) > 0)
        
        # 生成最终脉冲信号
        buy_condition = is_epu_extreme & is_epu_exhausted & is_bullish_confirm
        sell_condition = is_epu_extreme & is_epu_exhausted & is_bearish_confirm
        
        signal[buy_condition] = 1.0
        signal[sell_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, zscore_threshold={self.zscore_threshold})"