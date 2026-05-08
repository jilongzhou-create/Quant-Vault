import numpy as np
import pandas as pd

class VolCurveSteepeningReversalFactor:
    """波动率极值衰竭与收益率曲线走陡交叉因子 (volatility/nonlinear)

    逻辑: 恐慌极值与宏观预期的同频共振。当股市暴跌引发 VIX 达到极端极值时，可能仍处于主跌浪(纯粹的流动性抛售，接飞刀)。
          只有当 VIX 出现衰竭(开始边际回落)，且同期的 10年-2年 收益率曲线出现剧烈陡峭化 (Bull Steepening, 短端加速下行)，
          才证明市场逻辑已经从"流动性恐慌"转变为"定价美联储紧急降息/宽松"。
          此共振瞬间是做多美债(TLT)极高胜率的反转脉冲点。常态下零值休眠。
    数据: vixcls (VIX波动率指数), t10y2y (10年期减2年期利差)
    触发: VIX 63日 Z-Score > 2.5 且 衰竭回落 (diff < 0 且低于3日均线) + t10y2y 边际走陡 (diff > 0)
    输出: +1.0 看多美债, -1.0 看空美债 (VIX极端平静反弹且曲线走平/紧缩预期再起)
    """

    def __init__(self, zscore_window: int = 63, ma_window: int = 3):
        self.name = 'vol_curve_steepening_reversal_nonlinear'
        self.zscore_window = zscore_window
        self.ma_window = ma_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse) - 初始信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 保证时间序列顺序处理缺失值，避免前视偏差
        df = data[required_cols].ffill()
        
        vix = df['vixcls']
        t10y2y = df['t10y2y']
        
        # 铁律3: 边际变化 (Marginal Change Only) - 捕捉预期改变的瞬间
        vix_diff = vix.diff()
        t10y2y_diff = t10y2y.diff()
        
        # 铁律2: 二阶导数基准 (计算3日均线作为衰竭对比线)
        vix_ma = vix.rolling(window=self.ma_window).mean()
        
        # VIX Z-Score 绝对极值计算
        vix_mean = vix.rolling(window=self.zscore_window).mean()
        vix_std = vix.rolling(window=self.zscore_window).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        # --- 看多脉冲信号 (+1.0) ---
        # 1. 处于极度恐慌高位 (Z-Score > 2.5)
        # 2. 恐慌开始衰竭 (diff < 0 且 当前值 < 3日均值)
        # 3. 债券市场短端利率预期下行导致曲线突然变陡 (t10y2y.diff() > 0)
        long_extreme = (vix_z > 2.5)
        long_exhaustion = (vix_diff < 0) & (vix < vix_ma)
        long_curve_confirm = (t10y2y_diff > 0)
        
        long_cond = long_extreme & long_exhaustion & long_curve_confirm
        
        # --- 看空脉冲信号 (-1.0) ---
        # 1. 处于极端自满/低迷期 (Z-Score < -1.5)
        # 2. 波动率平静期被打破开始飙升 (diff > 0 且 当前值 > 3日均值)
        # 3. 曲线出现边际走平或倒挂加深，指向紧缩交易 (t10y2y.diff() < 0)
        short_extreme = (vix_z < -1.5)
        short_reversal = (vix_diff > 0) & (vix > vix_ma)
        short_curve_confirm = (t10y2y_diff < 0)
        
        short_cond = short_extreme & short_reversal & short_curve_confirm
        
        # 触发信号赋值
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(zscore_window={self.zscore_window}, ma_window={self.ma_window})"