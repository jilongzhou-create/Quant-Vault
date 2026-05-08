import numpy as np
import pandas as pd

class SyntheticMoveExhaustionFactor:
    """Synthetic MOVE (Bond Volatility) Exhaustion (volatility/options)

    逻辑: 针对债市波动率MOVE指数缺失的问题，遵循纯正FICC宏观框架，利用VIX(跨资产恐慌)与BBB信用利差波动率(微观信贷恐慌)合成“债市衍生品恐慌代理指标”。当该合成波动率指标狂飙至极值后开始回落时，标志着流动性冲击引发的“抛售一切(包括美债)换取现金”的去杠杆阶段结束。市场主线将瞬间切换至交易“衰退确认与美联储降息”，此时是做多美债(TLT)的极佳狙击点。非极端时期因子保持零值休眠。
    数据: vixcls (CBOE VIX), bamlc0a4cbbb (US Corporate BBB OAS)
    触发: 合成波动率的252日Z-Score > 2.5 (极值)，且当日环比下降且低于3日均值 (衰竭确认)
    输出: 脉冲信号 +1.0 (恐慌衰竭，做多TLT)
    """

    def __init__(self, zscore_window=252, vol_window=20, smooth_window=3, threshold=2.5):
        self.name = 'synthetic_move_exhaustion_volatility_options'
        self.zscore_window = zscore_window
        self.vol_window = vol_window
        self.smooth_window = smooth_window
        self.threshold = threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认全为0.0
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'bamlc0a4cbbb']
        if not all(col in data.columns for col in required_cols):
            return signal

        # 处理节假日导致的数据缺失，避免计算中断
        df = data[required_cols].ffill()

        # 1. 提取边际变化(铁律3): 计算信用利差的20日实际波动率，作为信贷期权隐含波动率的代理
        credit_vol = df['bamlc0a4cbbb'].diff().rolling(window=self.vol_window).std()

        # 2. 计算 VIX 和 Credit Vol 的独立 Z-Score 
        vix_mean = df['vixcls'].rolling(window=self.zscore_window).mean()
        vix_std = df['vixcls'].rolling(window=self.zscore_window).std()
        vix_z = (df['vixcls'] - vix_mean) / vix_std.replace(0, np.nan)

        cred_mean = credit_vol.rolling(window=self.zscore_window).mean()
        cred_std = credit_vol.rolling(window=self.zscore_window).std()
        cred_z = (credit_vol - cred_mean) / cred_std.replace(0, np.nan)

        # 3. 构建 Synthetic MOVE (无量纲的合成宏观/信用波动率冲击指数)
        synthetic_move = vix_z + cred_z

        # 计算合成指标的极值水位
        synth_mean = synthetic_move.rolling(window=self.zscore_window).mean()
        synth_std = synthetic_move.rolling(window=self.zscore_window).std()
        synth_zscore = (synthetic_move - synth_mean) / synth_std.replace(0, np.nan)

        # 4. 二阶导数条件(铁律2): 绝对禁止接飞刀！必须确认恐慌衰竭
        # 条件：动量小于0 (当日回落) 且 跌破短期均线 (确认趋势转折)
        is_exhausting = (synthetic_move.diff() < 0) & (synthetic_move < synthetic_move.rolling(window=self.smooth_window).mean())

        # 5. 极值条件: 处于罕见的恐慌高位
        is_extreme = synth_zscore > self.threshold

        # 6. 生成脉冲信号: 极值 + 衰竭 同时满足才触发
        trigger = is_extreme & is_exhausting
        
        # 赋值 +1.0 看多美债
        signal.loc[trigger] = 1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"SyntheticMoveExhaustionFactor(z_window={self.zscore_window}, thresh={self.threshold})"