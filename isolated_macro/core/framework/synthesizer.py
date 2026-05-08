#!/usr/bin/env python3
"""
Synthesizer - 卫星因子库与动态权重合成

核心机制: Rolling IC (Information Coefficient) 动态赋权
         + Conditional IC / Hit Rate 低频因子评估

原理:
  1. 底座信号 (core_signal) 只能解释黄金收益的一部分
  2. 残差 = 黄金实际收益 - 底座预期收益
  3. 每个卫星因子与残差的滚动相关性 = 该因子的"残差解释力"
  4. IC 为正且显著 → 赋予正权重; IC 不显著 → 权重衰减至 0 (自动淘汰)

低频因子评估 (v2):
  - Global Mean IC: 全局 IC 均值 (传统)
  - Conditional IC: 仅在 signal != 0 日的 IC (避免零信号稀释)
  - Hit Rate: 信号触发日, 因子方向与残差方向一致的概率
  - 准入规则: Global IC > 0.02 或 (Cond IC > 0.05 且 Hit Rate > 55% 且 Trigger > 0.1%)

合成公式:
  Total_Macro_Score = core_signal + Σ(sat_signal_i × sat_weight_i)
  最终 clip 到 [-1.0, 1.0]
"""

import numpy as np
import pandas as pd


class CreditPanicFactor:
    """
    信用恐慌因子 (脉冲型 + 持续性)

    逻辑: 信用利差 252 日 Z-Score > 2.0 → 信用恐慌爆发 → 避险买金
    数据: BAA10YM (穆迪 Baa 级企业债利差)
    输出: +1.0 (恐慌触发/持续中) 或 0.0 (无恐慌)

    纯粹性原则:
        - 只看信用利差本身, 不引入任何其他数据源
        - 过滤错误信号的任务交由系统架构完成:
          * CoreAnchor 在逆风期投出 -1 票压制敞口
          * DynamicSynthesizer 发现 IC 不显著时自动降权至 0
        - 因子逻辑一旦在 IS 通过, 代码即 FROZEN, 禁止基于 OOS 表现修改
    """

    def __init__(self, lookback: int = 252, zscore_threshold: float = 2.0,
                 regime_duration: int = 40):
        self.name = 'credit_panic'
        self.lookback = lookback
        self.zscore_threshold = zscore_threshold
        self.regime_duration = regime_duration

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        col = 'baa10ym' if 'baa10ym' in data.columns else 'bamlh0a0hym2'
        if col not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
        hy = data[col]
        if hy.isna().all() or hy.dropna().sum() == 0:
            return pd.Series(0.0, index=data.index, name=self.name)
        hy_mean = hy.rolling(window=self.lookback, min_periods=60).mean()
        hy_std = hy.rolling(window=self.lookback, min_periods=60).std()
        z = (hy - hy_mean) / (hy_std + 1e-9)
        raw_trigger = (z > self.zscore_threshold).astype(float)

        signal = pd.Series(0.0, index=data.index)
        countdown = 0
        for i in range(len(raw_trigger)):
            if raw_trigger.iloc[i] == 1.0:
                countdown = self.regime_duration
            if countdown > 0:
                signal.iloc[i] = 1.0
                countdown -= 1
        signal.name = self.name
        return signal

    def __repr__(self):
        return (f"CreditPanicFactor(lookback={self.lookback}, "
                f"threshold={self.zscore_threshold}, "
                f"regime_duration={self.regime_duration})")


class SgePremiumFactor:
    """
    东方溢价因子 (脉冲型)

    逻辑: 上海金溢价连续 N 天 > 阈值 → 东方抢筹 → 脱锚看多
    数据: SGE_Premium
    输出: +1.0 (脱锚触发) 或 0.0 (无脱锚)

    v2: 降低阈值和连续天数要求
        - v1 阈值 $20/5天, 触发率仅 0.42%, 统计样本不足
        - v2 阈值 $10/3天, 增加触发样本, 提升统计可靠性
        - $10 溢价仍显著高于正常水平(通常 <$5), 信号有效性不变
    """

    def __init__(self, premium_threshold: float = 10.0, consecutive_days: int = 3):
        self.name = 'sge_premium'
        self.premium_threshold = premium_threshold
        self.consecutive_days = consecutive_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        if 'sge_premium' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
        above = (data['sge_premium'] > self.premium_threshold).astype(int)
        consecutive = above.rolling(
            window=self.consecutive_days, min_periods=self.consecutive_days
        ).sum()
        signal = (consecutive >= self.consecutive_days).astype(float)
        signal.name = self.name
        return signal

    def __repr__(self):
        return (f"SgePremiumFactor(threshold={self.premium_threshold}, "
                f"consecutive={self.consecutive_days})")


class SatelliteFactor:
    """
    卫星因子容器

    包装一个因子实例, 附加动态权重计算逻辑。
    """

    def __init__(self, factor, ic_window: int = 60,
                 ic_threshold: float = 0.03, max_weight: float = 0.5,
                 weight_ema_span: int = 20):
        self.factor = factor
        self.ic_window = ic_window
        self.ic_threshold = ic_threshold
        self.max_weight = max_weight
        self.weight_ema_span = weight_ema_span
        self.name = factor.name

    def __repr__(self):
        return (f"SatelliteFactor(name='{self.name}', "
                f"ic_window={self.ic_window}, "
                f"max_weight={self.max_weight})")


class FactorResearcher:
    """
    因子研究员 - 低频事件型因子评估 (v3: 前向累积收益)

    核心改进:
      脉冲型因子(信用恐慌/东方溢价)是中期信号, 避险效应需要时间发酵。
      使用次日残差评估会导致 Hit Rate 被日频噪声稀释至 ~50%。
      改用前向 N 日累积残差评估, 捕捉因子的中期预测力。

    评估指标:
      - Global Mean IC: 全局 IC 均值 (基于前向累积残差)
      - Conditional IC: 仅在 signal != 0 日的 IC (真实预测力)
      - Hit Rate: 信号触发日, 前向累积残差方向与信号一致的概率
      - Trigger Rate: 信号非零的比例

    准入规则 (v3):
      Global Mean IC > 0.02
      或 (Conditional IC > 0.05 且 Hit Rate > 55% 且 Trigger Rate > 0.1%)
    """

    GLOBAL_IC_THRESHOLD = 0.02
    COND_IC_THRESHOLD = 0.05
    HIT_RATE_THRESHOLD = 0.55
    TRIGGER_RATE_THRESHOLD = 0.001
    FORWARD_PERIOD = 20

    @classmethod
    def evaluate_factor(cls, signal: pd.Series, residual: pd.Series,
                        name: str = '', forward_period: int = None) -> dict:
        fp = forward_period if forward_period is not None else cls.FORWARD_PERIOD

        forward_res = residual.rolling(fp).sum().shift(-fp)
        forward_res = forward_res.fillna(0)

        sig = signal.shift(1).fillna(0)
        res = forward_res

        valid = sig.notna() & res.notna()
        sig_v = sig[valid]
        res_v = res[valid]

        global_ic = sig_v.corr(res_v) if len(sig_v) > 20 else 0.0
        if pd.isna(global_ic) or np.isinf(global_ic):
            global_ic = 0.0

        active = sig_v != 0
        trigger_rate = active.mean()

        if active.sum() > 5:
            active_sig = sig_v[active]
            active_res = res_v[active]

            if active_sig.std() < 1e-9:
                cond_ic = active_res.mean() / (res_v.std() + 1e-9)
            else:
                cond_ic = active_sig.corr(active_res)
                if pd.isna(cond_ic) or np.isinf(cond_ic):
                    cond_ic = active_res.mean() / (res_v.std() + 1e-9)

            same_sign = (active_sig * active_res) > 0
            hit_rate = same_sign.mean()
            if pd.isna(hit_rate):
                hit_rate = 0.0
        else:
            cond_ic = 0.0
            hit_rate = 0.0

        accepted = (
            global_ic > cls.GLOBAL_IC_THRESHOLD
            or (
                cond_ic > cls.COND_IC_THRESHOLD
                and hit_rate > cls.HIT_RATE_THRESHOLD
                and trigger_rate > cls.TRIGGER_RATE_THRESHOLD
            )
        )

        return {
            'name': name,
            'global_ic': global_ic,
            'conditional_ic': cond_ic,
            'hit_rate': hit_rate,
            'trigger_rate': trigger_rate,
            'accepted': accepted,
            'forward_period': fp,
        }


class DynamicSynthesizer:
    """
    动态合成器 - 底座 + 卫星因子 Rolling IC 动态权重合成

    流程:
      1. 计算所有卫星因子信号
      2. 计算底座预期收益 (core_signal × 黄金收益)
      3. 计算残差 = 黄金收益 - 底座预期收益
      4. 计算前向 N 日累积残差 (捕捉中期预测力)
      5. 对每个卫星因子, 计算其信号与前向累积残差的 Rolling IC
      6. IC 显著 → 赋权; IC 不显著 → 权重衰减至 0
      7. 合成: total_score = core_signal + Σ(sat_signal_i × sat_weight_i)

    v2: 使用前向累积残差计算 IC
        - 日频 IC 对脉冲型因子不适用 (日频噪声稀释)
        - 前向累积残差捕捉因子的中期预测力
        - IC 阈值相应提高 (前向 IC 天然更大)
    """

    def __init__(self, ic_window: int = 60, ic_threshold: float = 0.05,
                 max_weight_per_factor: float = 0.5, weight_ema_span: int = 20,
                 forward_period: int = 20):
        self.ic_window = ic_window
        self.ic_threshold = ic_threshold
        self.max_weight_per_factor = max_weight_per_factor
        self.weight_ema_span = weight_ema_span
        self.forward_period = forward_period
        self.satellites: list = []

    def register(self, factor):
        sat = SatelliteFactor(
            factor=factor,
            ic_window=self.ic_window,
            ic_threshold=self.ic_threshold,
            max_weight=self.max_weight_per_factor,
            weight_ema_span=self.weight_ema_span,
        )
        self.satellites.append(sat)
        return self

    def evaluate_all(self, data: pd.DataFrame,
                     core_signal: pd.Series) -> list:
        """
        在 IS 期间评估所有卫星因子 (FactorResearcher 逻辑)

        Returns:
            list[dict]: 每个因子的评估结果
        """
        df = data.copy()
        gold_return = df['market_price'].pct_change()
        core_expected = core_signal.shift(1) * gold_return
        residual = (gold_return - core_expected.fillna(0)).fillna(0)

        results = []
        for sat in self.satellites:
            sig = sat.factor.calculate_signal(df)
            eval_result = FactorResearcher.evaluate_factor(
                sig, residual, name=sat.name
            )
            results.append(eval_result)

        return results

    def calculate(self, data: pd.DataFrame,
                  core_signal: pd.Series) -> pd.DataFrame:
        """
        执行信号合成

        Args:
            data: 原始数据 DataFrame
            core_signal: 底座信号 Series (来自 CoreMacroAnchor)

        Returns:
            DataFrame: 新增列
              - {sat_name}_signal: 每个卫星因子的信号
              - {sat_name}_ic: 每个卫星因子的 Rolling IC (基于前向累积残差)
              - {sat_name}_weight: 每个卫星因子的动态权重
              - total_score: 合成总分
              - residual: 底座残差
        """
        df = data.copy()
        df['core_signal'] = core_signal

        gold_return = df['market_price'].pct_change()
        core_expected = df['core_signal'].shift(1) * gold_return
        df['residual'] = gold_return - core_expected.fillna(0)

        forward_res = df['residual'].rolling(
            self.forward_period
        ).sum().shift(-self.forward_period)

        weighted_sum = pd.Series(0.0, index=df.index)

        for sat in self.satellites:
            sig_col = f'{sat.name}_signal'
            ic_col = f'{sat.name}_ic'
            wt_col = f'{sat.name}_weight'

            df[sig_col] = sat.factor.calculate_signal(df)

            ic_series = df[sig_col].shift(1).rolling(
                window=self.ic_window, min_periods=20
            ).corr(forward_res)
            ic_series = ic_series.replace([np.inf, -np.inf], 0.0)
            df[ic_col] = ic_series.fillna(0)

            raw_weight = ic_series.copy()
            raw_weight = raw_weight.where(raw_weight > self.ic_threshold, 0.0)
            raw_weight = raw_weight.clip(0, self.max_weight_per_factor)
            df[wt_col] = raw_weight.ewm(
                span=self.weight_ema_span, min_periods=1
            ).mean().fillna(0)

            weighted_sum += df[sig_col].fillna(0) * df[wt_col]

        df['total_score'] = (df['core_signal'].fillna(0) + weighted_sum).clip(-1.0, 1.0)

        return df

    def __repr__(self):
        names = [s.name for s in self.satellites]
        return (f"DynamicSynthesizer(satellites={names}, "
                f"ic_window={self.ic_window})")
