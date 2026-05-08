import numpy as np
import pandas as pd

class YieldCurvePivotNonlinearFactor:
    """收益率曲线政策预期极端跳跃与衰竭因子 (Policy Pivot Shock / Nonlinear)

    逻辑: 
    严格基于FICC经济学逻辑，结合短端利率(dgs2)与期限利差(t10y2y)的边际变化(动量)，捕捉美联储政策预期的突变脉冲与衰竭反转：
    1. 鸽派突变与鹰派衰竭 (做多TLT): 
       - 当短端利率暴跌且曲线急剧变陡(降息预期骤升)突破极值瞬间，触发顺势看多脉冲。
       - 当短端利率暴涨且曲线变平(加息恐慌)，但动量见顶回落且跌破3日均线时，极度悲观预期出尽，触发抄底看多脉冲。
    2. 鹰派突变与鸽派衰竭 (做空TLT):
       - 当短端利率暴涨且曲线急剧变平(加息预期骤升)突破极值瞬间，触发顺势看空脉冲。
       - 当短端利率暴跌且曲线变陡(降息狂热)，但动量见底反弹且突破3日均线时，极度乐观预期修正，触发逃顶看空脉冲。
    
    数据: dgs2, t10y2y
    触发: 5日变化量的63日Z-Score达到极端阈值，配合单日动量拐点与均线交叉进行二阶导数确认。
    输出: +1.0 (看多), -1.0 (看空), 严格的狙击手级脉冲信号。
    """

    def __init__(self):
        self.name = 'yield_curve_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，默认全0
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖列
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return signal

        # 填充缺失值，避免计算中断
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 铁律3: 边际变化 (绝对禁止使用绝对值，计算5日动量捕捉预期的边际突变)
        dgs2_diff = dgs2.diff(5)
        t10y2y_diff = t10y2y.diff(5)

        # 计算滚动63日(约一季度)的 Z-Score，定位短期政策预期的极端状态
        dgs2_mean = dgs2_diff.rolling(63).mean()
        dgs2_std = dgs2_diff.rolling(63).std() + 1e-6
        dgs2_z = (dgs2_diff - dgs2_mean) / dgs2_std

        t10y2y_mean = t10y2y_diff.rolling(63).mean()
        t10y2y_std = t10y2y_diff.rolling(63).std() + 1e-6
        t10y2y_z = (t10y2y_diff - t10y2y_mean) / t10y2y_std

        # 衰竭确认均线
        mean_3 = dgs2.rolling(3).mean()

        # 设定阈值: 兼顾极端的经济学含义与 5%~15% 的 Trigger Rate 铁律目标
        Z_DGS = 1.25
        Z_T10 = 0.75

        # ---------------------------------------------------------
        # 做多逻辑 (+1.0)
        
        # 1. 鸽派突变脉冲: 短端暴跌 + 曲线变陡，且刚刚跌破阈值的瞬间 (预期巨变的确认点)
        dovish_shock_long = (dgs2_z < -Z_DGS) & (dgs2_z.shift(1) >= -Z_DGS) & (t10y2y_z > Z_T10)

        # 2. 鹰派恐慌衰竭(铁律2): 短端暴涨 + 曲线变平极端拥挤，但动量停止恶化且跌破3日均线 (预期见顶)
        hawkish_exhaustion_long = (
            (dgs2_z > Z_DGS) & 
            (dgs2_diff.diff(1) < 0) & 
            (dgs2 < mean_3) & 
            (t10y2y_z < -Z_T10)
        )

        # ---------------------------------------------------------
        # 做空逻辑 (-1.0)
        
        # 1. 鹰派突变脉冲: 短端暴涨 + 曲线变平，且刚刚突破阈值的瞬间
        hawkish_shock_short = (dgs2_z > Z_DGS) & (dgs2_z.shift(1) <= Z_DGS) & (t10y2y_z < -Z_T10)

        # 2. 鸽派狂热衰竭(铁律2): 短端暴跌 + 曲线变陡极端拥挤，但动量见底反弹且突破3日均线 (利好出尽)
        dovish_exhaustion_short = (
            (dgs2_z < -Z_DGS) & 
            (dgs2_diff.diff(1) > 0) & 
            (dgs2 > mean_3) & 
            (t10y2y_z > Z_T10)
        )

        # ---------------------------------------------------------
        # 信号合成
        bull_cond = dovish_shock_long | hawkish_exhaustion_long
        bear_cond = hawkish_shock_short | dovish_exhaustion_short

        # 赋值并防御性处理极端重合的异常情况
        signal[bull_cond] = 1.0
        signal[bear_cond & ~bull_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"