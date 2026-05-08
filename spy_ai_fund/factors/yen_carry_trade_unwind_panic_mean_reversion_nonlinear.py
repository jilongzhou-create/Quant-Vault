import numpy as np
import pandas as pd

class CreditVixPanicReversalFactor:
    """Credit Vix Panic Reversal (panic_mean_reversion/nonlinear)

    逻辑: 结合高收益债信用利差(bamlh0a0hym2)与VIX(vixcls)，捕捉风险蔓延与恐慌衰竭。VIX反映短期的股市恐慌，信用利差反映中期的宏观资金面压力。
    数据: bamlh0a0hym2, vixcls
    输出: 恐慌见顶回落时输出 +1.0 (抄底)，信用走弱且VIX刚开始跳升时输出 -1.0 (避险)
    触发条件: 
      看多: 过去5天内VIX出现过1个月内的相对极值(Z > 1.2)，今日VIX较5日内高点回落超过5%且单日下跌，同时季度的信用利差确认了真实压力存在(Z > 0)。
      看空: 季度信用利差走阔(Z > 1.0)，VIX尚未极度恐慌(Z < 1.5)但单日跳升超5%(主跌浪初期)。
      预期 Trigger Rate 约 6%-10%。
    """

    def __init__(self):
        self.name = 'credit_vix_panic_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return signal

        vix = data['vixcls'].ffill()
        hy = data['bamlh0a0hym2'].ffill()

        # VIX反映短期情绪，使用21个交易日(约1个月)窗口寻找脉冲
        vix_mean = vix.rolling(window=21).mean()
        vix_std = vix.rolling(window=21).std().replace(0, 1e-5)
        vix_z = (vix - vix_mean) / vix_std

        # 信用利差反映中期基本面，使用63个交易日(约1个季度)窗口识别趋势
        hy_mean = hy.rolling(window=63).mean()
        hy_std = hy.rolling(window=63).std().replace(0, 1e-5)
        hy_z = (hy - hy_mean) / hy_std

        # --- 买入信号 (极端恐慌 + 衰竭) ---
        # 极值条件: 过去5天内VIX Z-Score曾大于1.2(处于月度相对高位)
        vix_recent_extreme = vix_z.rolling(window=5).max() > 1.2
        # 信用确认: 信用利差高于季度均值 (确认压力存在，非个股波动)
        credit_confirm = hy_z > 0.0
        # 恐慌衰竭脉冲: VIX较过去5日高点回落超过5%，且今日VIX为下跌
        vix_5d_max = vix.rolling(window=5).max()
        vix_drawdown = vix / vix_5d_max.replace(0, 1e-5)
        vix_exhaustion = (vix_drawdown < 0.95) & (vix.diff() < 0)
        # 零值休眠铁律：要求该衰竭现象昨天未发生，确保仅为单日脉冲
        buy_pulse = vix_exhaustion & ~(vix_exhaustion.shift(1).fillna(False))
        
        buy_signal = vix_recent_extreme & credit_confirm & buy_pulse

        # --- 卖出信号 (轻微恐慌/趋势恶化) ---
        # 信用恶化: 信用利差处于季度相对高位
        credit_worsening = hy_z > 1.0
        # VIX尚未进入极度恐慌 (防接飞刀，此时处于钝刀割肉或主跌浪初期)
        vix_not_extreme = vix_z < 1.5
        # VIX跳升脉冲: 今天VIX跳升超过5%
        vix_jump = (vix / vix.shift(1).replace(0, 1e-5)) > 1.05
        # 零值休眠铁律：要求昨天VIX没有跳升，防止连出负值
        vix_jump_pulse = vix_jump & ~(vix_jump.shift(1).fillna(False))

        sell_signal = credit_worsening & vix_not_extreme & vix_jump_pulse

        # --- 合成信号 ---
        signal[buy_signal] = 1.0
        signal[sell_signal] = -1.0

        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"