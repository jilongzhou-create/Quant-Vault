import numpy as np
import pandas as pd

class FlightToSafetyExhaustionFactor:
    """Flight to Safety Exhaustion (panic_mean_reversion/nonlinear)

    逻辑: 捕捉典型的股债双极值避险(Flight to Safety)模式及其衰竭瞬间。当恐慌急升且资金大幅涌入国债时输出看空脉冲；当恐慌见顶回落、资金流出避险资产时输出抄底买入脉冲。
    数据: vixcls (波动率), dgs10 (10年期美债收益率)
    输出: +1.0 表示避险情绪衰竭(强看多), -1.0 表示避险情绪爆发初期(看空), 常态 0.0
    触发条件: 抄底要求 VIX Z-Score > 1.5 且美债收益率大幅下降后双双出现反转；看空要求 VIX 单日跳涨 > 10% 且美债收益率单日骤降。预期 Trigger Rate 5%-15%。
    """

    def __init__(self, vix_z_window=252, dgs_momentum_window=21):
        self.name = 'flight_to_safety_exhaustion_nonlinear'
        self.vix_z_window = vix_z_window
        self.dgs_momentum_window = dgs_momentum_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查必要数据列
        if 'vixcls' not in data.columns or 'dgs10' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
        
        # 填充缺失值，避免无效计算
        vix = data['vixcls'].ffill()
        dgs10 = data['dgs10'].ffill()

        # 1. 计算 VIX 的极端极值状态 (252个交易日约一年)
        vix_mean = vix.rolling(window=self.vix_z_window, min_periods=63).mean()
        vix_std = vix.rolling(window=self.vix_z_window, min_periods=63).std()
        vix_zscore = (vix - vix_mean) / (vix_std + 1e-6)

        # 2. 计算国债收益率的中期动量 (反映逃向安全资产的资金趋势)
        # dgs10 下降代表国债价格上涨，资金疯狂涌入避险
        dgs10_mom = dgs10.diff(self.dgs_momentum_window)
        
        # 3. 边际变化率 (二阶导数转折核心)
        vix_diff_1d = vix.diff(1)
        vix_pct_1d = vix.pct_change(1)
        dgs10_diff_1d = dgs10.diff(1)

        # 初始化信号序列
        signal = pd.Series(0.0, index=data.index)

        # 【买入条件】: 极端恐慌 + 避险衰竭 (捕捉脉冲反转点)
        # 1. VIX 处于历史高位极值 (Z-Score > 1.5)
        # 2. 过去一个月资金大幅避险涌入国债 (收益率下降 > 15bps)
        # 3. 衰竭反转日：VIX 恐慌值回落，且美债收益率停止下降开始反弹
        long_pulse = (vix_zscore > 1.5) & \
                     (dgs10_mom < -0.15) & \
                     (vix_diff_1d < 0) & \
                     (dgs10_diff_1d > 0)

        # 【看空条件】: 恐慌初步爆发 (恶化脉冲)
        # 1. VIX 单日跳涨超过 10% (突发恐慌)
        # 2. 单日美债收益率骤降 > 5bps (资金紧急避险)
        # 3. VIX 并未处于极端高位，表明刚起步，杜绝跌到底部接飞刀看空
        short_pulse = (vix_pct_1d > 0.10) & \
                      (dgs10_diff_1d < -0.05) & \
                      (vix_zscore < 1.5)

        # 合并信号
        signal.loc[long_pulse] = 1.0
        signal.loc[short_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(vix_z_window={self.vix_z_window}, dgs_momentum_window={self.dgs_momentum_window})"