import numpy as np
import pandas as pd

class PanicSpreadReversionFactor:
    """恐慌均值回归因子 (panic_mean_reversion/nonlinear)

    逻辑: 当VIX和高收益债利差均飙升至极高位时(Z-Score>1.5), 绝对不能直接买入(防接飞刀)。只有当VIX见顶回落(今日下跌且低于前3日均值)时, 确认恐慌衰竭, 输出脉冲看多(+1.0)。反之, 当波动率和利差在中高位且连续两日上升时, 视为钝刀割肉的流动性恶化, 输出脉冲看空(-1.0)。
    数据: [vixcls, bamlh0a0hym2]
    输出: +1.0 表示恐慌极值后的衰竭抄底, -1.0 表示恐慌温水煮青蛙式恶化, 0.0 表示常态
    触发条件: VIX与利差的Z-Score极值与各自的动量(二阶导数)非线性交叉, 预期Trigger Rate控制在 5%-15% 之间
    """

    def __init__(self, vix_z_long=1.5, spread_z_long=1.0, vix_z_short=0.5, spread_z_short=0.5, pulse_hold_days=3):
        self.name = 'panic_spread_reversion_pulse'
        self.vix_z_long = vix_z_long
        self.spread_z_long = spread_z_long
        self.vix_z_short = vix_z_short
        self.spread_z_short = spread_z_short
        self.pulse_hold_days = pulse_hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据是否完全存在
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            signal.name = self.name
            return signal
            
        vix = data['vixcls'].ffill()
        spread = data['bamlh0a0hym2'].ffill()
        
        # 计算 252个交易日的滚动 Z-Score (反映宏观级别的历史相对高低水位)
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        spread_mean = spread.rolling(window=252, min_periods=60).mean()
        spread_std = spread.rolling(window=252, min_periods=60).std()
        spread_z = (spread - spread_mean) / spread_std.replace(0, np.nan)
        
        # -------------------------------------------------------------
        # 多头逻辑 (强力抄底脉冲): 极度恐慌 + 恐慌见顶衰竭 (严格防接飞刀)
        # -------------------------------------------------------------
        # 二阶衰竭条件: 今日VIX下跌, 且必须低于过去3天的均值 (确认真正的动能反转)
        vix_exhaustion = (vix.diff(1) < 0) & (vix < vix.shift(1).rolling(window=3).mean())
        long_cond = (vix_z > self.vix_z_long) & (spread_z > self.spread_z_long) & vix_exhaustion
        
        # -------------------------------------------------------------
        # 空头逻辑 (趋势看空脉冲): 轻/中度恐慌 + 风险资产流动性缓慢恶化
        # -------------------------------------------------------------
        # 恶化条件: VIX 和 信用利差连续两天同时上升 (温水煮青蛙, 市场持续承压)
        vix_worsen = (vix.diff(1) > 0) & (vix.shift(1).diff(1) > 0)
        spread_worsen = (spread.diff(1) > 0) & (spread.shift(1).diff(1) > 0)
        short_cond = (vix_z > self.vix_z_short) & (vix_z <= self.vix_z_long) & (spread_z > self.spread_z_short) & vix_worsen & spread_worsen
        
        # 转换为瞬时触发的脉冲 (只在状态转变的瞬间触发)
        long_pulse = long_cond & ~long_cond.shift(1).fillna(False)
        short_pulse = short_cond & ~short_cond.shift(1).fillna(False)
        
        # 将脉冲维持极短的休眠窗口期(如3天)，随后立刻休眠归 0.0，达成 5%-15% 的目标Trigger Rate
        long_signal = long_pulse.astype(float).rolling(window=self.pulse_hold_days, min_periods=1).max()
        short_signal = short_pulse.astype(float).rolling(window=self.pulse_hold_days, min_periods=1).max()
        
        # 整合信号 (-1.0 到 1.0)
        signal[long_signal > 0] = 1.0
        signal[short_signal > 0] = -1.0
        
        # 若出现极小概率的多空重叠, 在长牛美股市场以多头抄底衰竭优先
        conflict = (long_signal > 0) & (short_signal > 0)
        signal[conflict] = 1.0
        
        # 填充可能因NaN产生的空值
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(vix_z_long={self.vix_z_long}, spread_z_long={self.spread_z_long})"