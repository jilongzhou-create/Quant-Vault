#!/usr/bin/env python3
"""
Core Anchor - TLT 美债宏观底座因子 (v13 - Yield Shock Z-Score + Value)

彻底摒弃均线交叉，升级为机构级收益率波动率冲击模型。
SMA21/SMA63 过于敏感导致 44% 时间空仓，严重踏空温和牛市。

三象限法则:
  象限 1: 恐慌危机 (Crisis) - 信用利差 > SMA126 → 1.0 满仓避险
  象限 2: 紧缩冲击 (Tightening Shock Veto) - DGS10 Z-Score > 1.5 → 0.0
    收益率 1年期 Z-Score > 1.5σ = 真实宏观紧缩狂飙 = 剥夺估值开仓权
    日常反弹噪音被 1.5σ 阈值完美过滤，大幅解放空仓时间
  象限 3: 温和常态 (Normal Value) - TIPS Z-Score 估值吃息
    只有收益率不在极端冲击时，估值引擎才全力工作

优先级: 恐慌(1.0) > 紧缩冲击(0.0) > 常态估值
"""

import numpy as np
import pandas as pd


class TltCoreAnchor:
    """
    TLT 宏观底座 - Yield Shock Z-Score + Value (v13)
    """

    def __init__(self):
        self.name = 'tlt_core_anchor'

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # ── 1. 恐慌避险锚 (Crisis) ──
        df['spread_sma126'] = df['BAMLH0A0HYM2'].rolling(
            window=126, min_periods=63
        ).mean()
        df['is_panic'] = df['BAMLH0A0HYM2'] > df['spread_sma126']

        # ── 2. 紧缩冲击雷达 (Tightening Shock Veto) ──
        # DGS10 1年期 Z-Score: 衡量收益率偏离过去1年均值的暴力程度
        df['dgs10_mean252'] = df['DGS10'].rolling(window=252, min_periods=126).mean()
        df['dgs10_std252'] = df['DGS10'].rolling(window=252, min_periods=126).std()
        df['dgs10_zscore'] = (
            (df['DGS10'] - df['dgs10_mean252']) / (df['dgs10_std252'] + 1e-6)
        )
        df['dgs10_zscore'] = df['dgs10_zscore'].fillna(0.0)
        # 只有收益率向上狂飙超过 1.5σ 才认定为真实宏观紧缩
        df['is_tightening_shock'] = df['dgs10_zscore'] > 1.5

        # ── 3. 温和常态引擎 (Normal Value) - TIPS Z-Score ──
        df['tips_mean_3y'] = df['DFII10'].rolling(window=756, min_periods=126).mean()
        df['tips_std_3y'] = df['DFII10'].rolling(window=756, min_periods=126).std()
        df['tips_zscore'] = (
            (df['DFII10'] - df['tips_mean_3y']) / (df['tips_std_3y'] + 1e-6)
        )
        df['tips_zscore'] = df['tips_zscore'].fillna(0.0)
        df['normal_carry'] = np.clip(0.5 + (df['tips_zscore'] * 0.25), 0.0, 1.0)

        # ── 宏观象限路由 ──
        conditions = [df['is_panic'], df['is_tightening_shock']]
        choices = [1.0, 0.0]
        df['tlt_core_signal'] = np.select(conditions, choices, default=df['normal_carry'])

        # ── NaN 容错 ──
        fill_cols = ['is_panic', 'is_tightening_shock', 'dgs10_zscore',
                     'tips_zscore', 'normal_carry', 'tlt_core_signal']
        for col in fill_cols:
            df[col] = df[col].fillna(0.0)

        return df

    def __repr__(self):
        return "TltCoreAnchor(mode=Yield_Shock_ZScore_Value_v13)"
