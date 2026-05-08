import numpy as np
import pandas as pd

class YieldCurvePivotShockFactor:
    """Yield Curve Pivot Shock Pulse (policy_pivot/nonlinear)

    逻辑: 捕捉美联储鸽派/鹰派预期瞬间剧变(流动性冲量反转)的脉冲信号。当短端利率(DGS2)在5天内出现具有经济意义的剧烈下行(>15bp，相当于市场抢跑半次25bp降息)且期限利差(T10Y2Y)快速走阔(>10bp)，且两者的边际变化在近半年(126天)属于超过1.5倍标准差的异常尾部事件时，视为市场确认政策转向的 Bull Steepening 瞬间，触发看多脉冲；反之，短端急速飙升且曲线被过度压平(Bear Flattening)则触发看空脉冲。
    数据: dgs2 (2年期美债收益率), t10y2y (10年期与2年期利差)
    输出: +1.0 表示鸽派转向预期导致收益率曲线急速变陡(看多美股), -1.0 表示鹰派紧缩预期导致曲线急速变平(看空美股), 其他常态下为 0.0。
    触发条件: 利率变动动量同时跨越 1.5倍 Z-Score 极值阈值 及 基点绝对阈值(15bp/10bp) 的突破首日，预期 Trigger Rate 控制在 5%-15% 之间的狙击手频率。
    """

    def __init__(self, diff_window=5, z_window=126, bp_dgs2=0.15, bp_t10y2y=0.10, z_threshold=1.5):
        self.name = 'yield_curve_pivot_shock_pulse'
        self.diff_window = diff_window
        self.z_window = z_window
        self.bp_dgs2 = bp_dgs2
        self.bp_t10y2y = bp_t10y2y
        self.z_threshold = z_threshold

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 常态下必须默认返回 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 纯粹的本领域数据依赖，缺少数据则不触发
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 1. 严格使用边际变化(.diff)，绝对禁止使用绝对水位
        # 计算 5 个交易日的动量变化
        dgs2_diff = dgs2.diff(self.diff_window)
        t10y2y_diff = t10y2y.diff(self.diff_window)
        
        # 2. 计算动态基准 Z-Score (使用过去半年 126 个交易日作为宏观环境窗口)
        dgs2_mean = dgs2_diff.rolling(window=self.z_window, min_periods=21).mean()
        dgs2_std = dgs2_diff.rolling(window=self.z_window, min_periods=21).std()
        t10y2y_mean = t10y2y_diff.rolling(window=self.z_window, min_periods=21).mean()
        t10y2y_std = t10y2y_diff.rolling(window=self.z_window, min_periods=21).std()
        
        dgs2_z = (dgs2_diff - dgs2_mean) / (dgs2_std + 1e-8)
        t10y2y_z = (t10y2y_diff - t10y2y_mean) / (t10y2y_std + 1e-8)
        
        # 3. 非线性特征交叉
        # 多头脉冲: 极端的 Bull Steepening (短端利率骤降 + 曲线变陡)
        buy_cond = (
            (dgs2_diff <= -self.bp_dgs2) &        # 短端剧烈下行至少 15bp
            (t10y2y_diff >= self.bp_t10y2y) &     # 期限利差急剧走阔至少 10bp
            (dgs2_z <= -self.z_threshold) &       # 下行幅度达到统计学 1.5 倍标准差
            (t10y2y_z >= self.z_threshold)        # 变陡幅度达到统计学 1.5 倍标准差
        )
                   
        # 空头脉冲: 极端的 Bear Flattening (短端利率骤升 + 曲线变平/倒挂加深)
        sell_cond = (
            (dgs2_diff >= self.bp_dgs2) &         # 短端剧烈飙升至少 15bp
            (t10y2y_diff <= -self.bp_t10y2y) &    # 期限利差急剧收窄至少 10bp
            (dgs2_z >= self.z_threshold) &        # 飙升幅度达到统计学 1.5 倍标准差
            (t10y2y_z <= -self.z_threshold)       # 收窄幅度达到统计学 1.5 倍标准差
        )
                    
        # 4. 零值休眠铁律的最后防线: 提取"跨越临界值瞬间"的脉冲事件
        # 确保只在条件达成的第一天触发，消除连续天数的重复信号
        buy_pulse = buy_cond & ~buy_cond.shift(1).fillna(False)
        sell_pulse = sell_cond & ~sell_cond.shift(1).fillna(False)
        
        signal.loc[buy_pulse] = 1.0
        signal.loc[sell_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(diff_window={self.diff_window}, z_window={self.z_window})"