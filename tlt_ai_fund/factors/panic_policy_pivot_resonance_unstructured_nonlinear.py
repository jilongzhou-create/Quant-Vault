import numpy as np
import pandas as pd

class PanicPolicyPivotResonanceFactor:
    """恐慌与政策预期共振转向因子 (Policy Pivot Shock / Nonlinear Resonance)

    逻辑: 这是一个捕捉美联储政策预期突变与市场恐慌共振衰竭的非线性脉冲因子。当金融系统处于极端恐慌(VIX极高)且情绪刚开始见顶回落时，若同时短端政策预期利率(dgs2)急剧暴跌，且收益率曲线(t10y2y)发生剧烈的看涨陡峭化(Bull Steepening)，预示美联储即将超预期降息救市，产生强力看多美债(TLT)的趋势脉冲；反之，若在极度安逸状态下加息预期猛烈抬头，则看空美债。因子严格遵循零值休眠与二阶导数衰竭铁律。
    数据: vixcls, dgs2, t10y2y
    触发: VIX 90日Z-Score极值(>2.0或<-1.5)且伴随均值回落的二阶衰竭 + dgs2 5日动量Z-Score极值(<-2.0或>2.0) + t10y2y 5日动量Z-Score变陡/平坦化共振。
    输出: 满足极端非线性共振且符合衰竭条件时，输出 +1.0 或 -1.0 的脉冲信号，其余时间休眠保持为 0.0。
    """

    def __init__(self, window: int = 90, diff_days: int = 5):
        self.name = 'panic_policy_pivot_resonance'
        self.window = window       # 宏观基准评估滚动窗口 (约一个季度)
        self.diff_days = diff_days # 捕捉动量突变的天数 (约一个交易周)

    def _calc_zscore(self, s: pd.Series) -> pd.Series:
        """计算滚动Z-Score，体现当前值偏离近期常态的极端程度"""
        roll_mean = s.rolling(self.window).mean()
        roll_std = s.rolling(self.window).std().replace(0.0, 1e-5)
        return (s - roll_mean) / roll_std

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始值为绝对的 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 提取必须字段，缺失则返回全0
        req_cols = ['vixcls', 'dgs2', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        df = data[req_cols].ffill()
        
        # --- 维度1: 恐慌指标状态与二阶衰竭 (遵守二阶导数铁律) ---
        vix = df['vixcls']
        vix_z = self._calc_zscore(vix)
        # 绝对禁止直接追高买入飞刀，必须等极端情绪开始回落/反弹
        vix_exhaustion = vix < vix.rolling(3).mean()  # 恐慌见顶开始回落
        vix_rebound = vix > vix.rolling(3).mean()     # 极度安逸被打破，恐慌抬头
        
        # --- 维度2: 政策预期前瞻动量 (遵守边际变化铁律) ---
        # 绝对禁止使用绝对水位，必须使用边际变化量寻找预期骤变的突变点
        dgs2_diff = df['dgs2'].diff(self.diff_days)
        dgs2_diff_z = self._calc_zscore(dgs2_diff)
        
        # --- 维度3: 收益率曲线形态动量 (遵守边际变化铁律) ---
        # 不要关注倒挂的持续状态，而是关注其急剧变陡峭或急剧平坦化的动能脉冲
        t10y2y_diff = df['t10y2y'].diff(self.diff_days)
        t10y2y_diff_z = self._calc_zscore(t10y2y_diff)
        
        # 构造非线性交叉脉冲触发条件 (方法C)
        
        # 多头脉冲(1.0): 恐慌极度爆表且刚见顶回落 + dgs2暴跌(极强降息预期) + t10y2y加速变陡(Bull Steepening)
        long_cond = (
            (vix_z > 2.0) & vix_exhaustion &
            (dgs2_diff_z < -2.0) &
            (t10y2y_diff_z > 1.0)
        )
        
        # 空头脉冲(-1.0): 极度安逸状态结束开始反弹 + dgs2暴涨(超预期加息冲击) + t10y2y急剧平坦化/深度倒挂
        short_cond = (
            (vix_z < -1.5) & vix_rebound &
            (dgs2_diff_z > 2.0) &
            (t10y2y_diff_z < -1.0)
        )
        
        # 只有在非线性共振触发极值脉冲的瞬时点才输出方向信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, diff_days={self.diff_days})"