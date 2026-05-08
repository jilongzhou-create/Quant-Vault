import numpy as np
import pandas as pd

class RateSteepeningCreditPulseFactor:
    """Rate Steepening Credit Pulse (policy_pivot/nonlinear)

    逻辑: 捕捉市场对美联储政策转向预期的极短窗口抢跑定价。当短端利率(DGS2)剧烈下行导致收益率曲线(T10Y2Y)急剧变陡(Bull Steepening)，且高收益债信用利差(HY OAS)停止恶化或开始收窄时，表明宽松预期确立且无系统性信用衰退风险，触发抄底脉冲(防接飞刀)；反之，短端利率急升导致曲线平坦化且信用恶化时，触发看空脉冲。
    数据: dgs2, t10y2y, bamlh0a0hym2
    输出: [-1.0, 1.0] 的脉冲信号，正值看多美股，负值看空美股
    触发条件: DGS2 5日下跌 > 12bps，且 T10Y2Y 5日走阔 > 8bps，配合信用利差3日动量<=0。预期 Trigger Rate 5%-12%。
    """

    def __init__(self, dgs2_diff_window=5, t10y2y_diff_window=5, credit_diff_window=3, rate_threshold=0.12, curve_threshold=0.08):
        self.name = 'rate_steepening_credit_pulse'
        # 窗口期
        self.dgs2_diff_window = dgs2_diff_window
        self.t10y2y_diff_window = t10y2y_diff_window
        self.credit_diff_window = credit_diff_window
        # 变动阈值 (百分比绝对值，如 0.12 = 12 bps)
        self.rate_threshold = rate_threshold
        self.curve_threshold = curve_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查是否有所需的数据字段
        required_cols = ['dgs2', 't10y2y', 'bamlh0a0hym2']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0.0, index=data.index, name=self.name)

        # 提取数据并沿时间轴前向填充，防止日内更新频率不同导致的空值干扰
        df = data[required_cols].ffill()

        # 计算边际变化(二阶导逻辑)，绝对禁止使用绝对值作为触发条件
        dgs2_momentum = df['dgs2'].diff(self.dgs2_diff_window)
        curve_momentum = df['t10y2y'].diff(self.t10y2y_diff_window)
        credit_momentum = df['bamlh0a0hym2'].diff(self.credit_diff_window)

        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=df.index, name=self.name)

        # ----------------------------------------------------------------------
        # 多头条件：Bull Steepening (抢跑宽松) + 信用风险衰竭 (未恶化)
        # DGS2 急跌 && 曲线急陡 && 信用利差未走阔(防接流动性危机飞刀)
        # ----------------------------------------------------------------------
        buy_cond = (
            (dgs2_momentum < -self.rate_threshold) & 
            (curve_momentum > self.curve_threshold) & 
            (credit_momentum <= 0.0)
        )

        # ----------------------------------------------------------------------
        # 空头条件：Bear Flattening (紧缩超预期) + 信用风险升温
        # DGS2 急升 && 曲线急平/倒挂加深 && 信用利差开始走阔
        # ----------------------------------------------------------------------
        sell_cond = (
            (dgs2_momentum > self.rate_threshold) & 
            (curve_momentum < -self.curve_threshold) & 
            (credit_momentum > 0.0)
        )

        # 脉冲触发
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(rate_threshold={self.rate_threshold}, curve_threshold={self.curve_threshold})"